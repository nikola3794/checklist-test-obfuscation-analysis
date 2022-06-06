#!/usr/bin/env python3
"""
In order to make this script, the validation.py script was taken and modified from:
https://github.com/rwightman/pytorch-image-models
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from torch.nn import functional as F

from datetime import datetime
import random
import string

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

parser.add_argument('--ae-step-list', type=float, default=[1.0, 1.0], nargs='+')
parser.add_argument('--ae-eps-list', type=float, default=[2.0, 4.0], nargs='+')
parser.add_argument('--ae-n-steps', type=int, default=5)
parser.add_argument('--ae-n-expectation-samples', type=int, default=[1, 3], nargs='+')
parser.add_argument('--ae-n-restarts', type=int, default=2)
parser.add_argument('--ae-random-init', type=bool, default=True)

parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset', default='unspecified')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='unspecified',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=100, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='unpecified', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='/home/nipopovic/Projects/euler/work_specta/experiment_logs/_tmp', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])        
    _logger.info(
        f'Model {args.model} created, param count: {param_count} ({(float(param_count)/(10.0**6)):.1f} M)'
    )

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data_dir, name=args.dataset, split=args.split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
        persistent_workers=False)

    # Normalization parameters
    global mean_tensor
    global std_tensor
    mean_tensor = torch.tensor([x for x in data_config['mean']]).cuda().view(1, 3, 1, 1)
    std_tensor = torch.tensor([x for x in data_config['std']]).cuda().view(1, 3, 1, 1)

    N_samples = [1]
    eval_i = 0
    assert len(args.ae_eps_list) == len(args.ae_step_list)
    for n_expectation_samples in args.ae_n_expectation_samples:
        for eps, step_size in zip(args.ae_eps_list, args.ae_step_list):
            _logger.info(f'eps={eps}/255')
            _logger.info(f'step-size={step_size}/255')
            _logger.info(f'n-expectation-samples={n_expectation_samples}')
            _logger.info("------------------------------")
        
            # Adjust step size to maximum color intensity of 255
            eps = eps / 255.0
            step_size = step_size / 255.0

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = {}
            top5 = {}
            for N_smpl in N_samples:
                top1[N_smpl] = AverageMeter()
                top5[N_smpl] = AverageMeter()

            model.eval()
            end = time.time()

            for batch_idx, (input, target) in enumerate(loader):
                if args.no_prefetcher:
                    target = target.cuda()
                    input = input.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                if eps != 0.0:
                    # Denormalize data     
                    input = input.mul_(std_tensor).add_(mean_tensor)

                    orig_input = input.clone()

                    # Number of random restarts when crafting the adversary example
                    for i_res in range(args.ae_n_restarts):
                        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
                        input = orig_input + randn
                        input.clamp_(0, 1.0)

                        # Number of PGD adversary attack steps
                        for i_step in range(args.ae_n_steps):
                            
                            # Expectation approximation of the gradient for stochastic pipelines
                            for i_exp in range(n_expectation_samples):
                                invar = torch.autograd.Variable(input, requires_grad=True)
                                in1 = invar - mean_tensor
                                in1.div_(std_tensor)
                                output = model(in1)
                                ascend_loss = criterion(output, target)
                                if i_exp == 0:
                                    ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
                                else:
                                    ascend_grad += torch.autograd.grad(ascend_loss, invar)[0]
                            #ascend_grad /= (i_exp+1) # Not necessary for PGD since sign is taken

                            # Apply adversarial purturbation
                            pert = step_size*torch.sign(ascend_grad)
                            input += pert.data
                            input = torch.max(orig_input-eps, input)
                            input = torch.min(orig_input+eps, input)
                            input.clamp_(0, 1.0)

                        input.sub_(mean_tensor).div_(std_tensor)
                        with torch.no_grad(): 
                            if i_res == 0: 
                                final_input = input.clone()
                            else:                     
                                output = model(input)
                                I = output.max(1)[1] != target
                                final_input[I] = input[I]
                    
                    input = final_input
                
                with torch.no_grad():
                    # compute output
                    with amp_autocast():
                        for smpl_i in range(N_samples[-1]):
                            output_tmp = model(input)
                            output_tmp_softmax = F.softmax((output_tmp), dim=1)
                            if smpl_i == 0:
                                output = output_tmp
                                output_softmax = output_tmp_softmax
                            else:
                                output += output_tmp
                                output_softmax += output_tmp_softmax
                            for N_smpl in N_samples:
                                if (smpl_i+1) == N_smpl:
                                    # Dont need to divide with Nsince accuracy metric will pick the maximal values
                                    acc1, acc5 = accuracy(output_softmax.detach(), target, topk=(1, 5))
                                    top1[N_smpl].update(acc1.item(), input.size(0))
                                    top5[N_smpl].update(acc5.item(), input.size(0))

                    if valid_labels is not None:
                        output = output[:, valid_labels]
                    loss = criterion(output_tmp, target)

                    if real_labels is not None:
                        real_labels.add_result(output)

                    # measure accuracy and record loss
                    losses.update(loss.item(), input.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if batch_idx % args.log_freq == 0:
                        log_msg = ''
                        for log_j, N_smpl in enumerate(N_samples):
                            if log_j == 2:
                                break
                            if log_j == 0:
                                _logger.info(
                                    'PGD Test: [{0:>4d}/{1}]  '
                                    'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s, {rate_avg:>7.2f}/s)  '
                                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                                        batch_idx, len(loader), batch_time=batch_time,
                                        rate_avg=input.size(0) / batch_time.avg, loss=losses)
                                    )
                            log_msg += f'Acc@1(N={N_smpl}):{top1[N_smpl].val:>7.2f} ({top1[N_smpl].avg:>7.2f}) |'
                            log_msg += f'Acc@5(N={N_smpl}):{top5[N_smpl].val:>7.2f} ({top5[N_smpl].avg:>7.2f}) \n'
                        _logger.info(log_msg)

            if real_labels is not None:
                raise NotImplementedError
                # real labels mode replaces topk values at the end
                #top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)


            # Log to .csv file
            header = [
                'model',

                'ae-eps',
                'ae-step-size',
                'ae-n-steps',
                'ae-n-restarts',
                'ae-n-expectation-samples',
            ]
            header.extend([f'acc1 (N={N_samples[0]})', f'acc1 (N={N_samples[-1]})'])
            header.extend([f'acc1 (N={x})' for x in N_samples])
            header.extend([f'acc5 (N={x})' for x in N_samples])
            header.append('exp-dir')
            data = [
                args.model,

                f'{eps*255.0}/255',
                f'{step_size*255.0}/255',
                args.ae_n_steps,
                args.ae_n_restarts,
                n_expectation_samples,
            ]
            data.extend([f'{top1[N_samples[0]].avg:.2f}', f'{top1[N_samples[-1]].avg:.2f}'])
            data.extend([f'{top1[x].avg:.2f}' for x in N_samples])
            data.extend([f'{top5[x].avg:.2f}' for x in N_samples])
            data.append(args.checkpoint)
            if eval_i == 0:
                exp_dir = os.path.dirname(args.checkpoint)
                if not os.path.isdir(exp_dir):
                    assert os.path.isdir(os.path.dirname(exp_dir))
                    os.mkdir(exp_dir)
                save_path = os.path.join(exp_dir, f'eps_{eps*255.0}_adversarial_robustness_evaluation.csv')
                mode = 'w'
                if os.path.isfile(save_path):
                    mode = 'a'
                with open(save_path, mode) as fh:
                    writer = csv.writer(fh)
                    if mode == 'w':
                        writer.writerow(header)
                    writer.writerow(data)
            else:
                with open(save_path, 'a') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(data)
            
            _logger.info('Finally.....')
            log_msg = ''
            for log_j, N_smpl in enumerate(N_samples):
                if log_j == 0:
                    _logger.info(
                        'PGD Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                            batch_idx, len(loader), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg, loss=losses)
                        )
                log_msg += f'Acc@1(N={N_smpl}):{top1[N_smpl].avg:>7.2f} |'
                log_msg += f'Acc@5(N={N_smpl}):{top5[N_smpl].avg:>7.2f} \n'
            _logger.info(log_msg)
            _logger.info(' ')
            _logger.info(' ')
            eval_i += 1

    return -1


def main():
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]   

    timestamp_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = os.path.dirname(args.checkpoint)
    rnd_number=''.join(random.choice(string.ascii_letters) for i in range(4))
    log_path = os.path.join(log_dir , f"eval_adversarial_{timestamp_str}_{rnd_number}.txt")
    setup_default_logging(log_path=log_path)
    # TODO <-------------------- Adapt

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
