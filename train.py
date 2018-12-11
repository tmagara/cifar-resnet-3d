import argparse
import chainer
import math
import numpy
from chainer.backends.cuda import get_device_from_id
from chainer.training import extensions

import net_3d_basic


class CroppingDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, crop_size, erase_size, erase_count, do_flip):
        self.dataset = dataset
        self.crop_size = crop_size
        self.erase_size = erase_size
        self.erase_count = erase_count
        self.do_flip = do_flip

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        x, t = self.dataset[i]

        xp = numpy

        mask = xp.ones((1,) + x.shape[1:], x.dtype)
        x = xp.concatenate((x, mask), 0)

        if self.crop_size > 0 and xp.random.uniform() > 0.5:
            x = random_crop(xp, x, self.crop_size)

        erase_count = xp.random.random_integers(0, self.erase_count)
        if erase_count > 0:
            x[:] *= mask_for_erase(xp, x.shape, x.dtype, self.erase_size, erase_count)

        if self.do_flip and numpy.random.uniform() > 0.5:
            x = xp.flip(x, 2)

        r = xp.random.uniform(size=(x.shape[0] - 1,) + x.shape[1:])
        x[:-1] = x[-1:] * x[:-1] + (1 - x[-1:]) * r

        return x, t


def random_crop(xp, x, crop_size):
    C, H, W = x.shape

    x = xp.pad(x, [(0, 0), (crop_size, crop_size), (crop_size, crop_size)], mode='constant', constant_values=0)

    dest = xp.zeros((C, H, W), x.dtype)
    top = xp.random.random_integers(0, crop_size * 2)
    left = xp.random.random_integers(0, crop_size * 2)
    dest[:] = x[:, top:top + H, left:left + W]
    return dest


def mask_for_erase(xp, x_shape, x_dtype, erase_size, repeat):
    C, H, W = x_shape

    x = xp.ones((1, H, W), x_dtype)
    for _ in range(repeat):
        left = xp.random.random_integers(0, W - erase_size)
        top = xp.random.random_integers(0, H - erase_size)
        x[:, left:left + erase_size, top:top + erase_size] = 0

    return x


def main():
    parser = argparse.ArgumentParser(description='3D ResNet on CIFAR')
    parser.add_argument('--dataset', '-d', default='cifar-10', choices=('cifar-10', 'cifar-100'),
                        help='The dataset name')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--learningrate', '-l', type=float, default=0.001,
                        help='alpha value for Adam')
    parser.add_argument('--weightdecay', '-w', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    if args.dataset == 'cifar-10':
        class_labels = 10
        train, test = chainer.datasets.get_cifar10()
    else:
        class_labels = 100
        train, test = chainer.datasets.get_cifar100()

    train = CroppingDataset(train, 4, 8, 1, True)
    test = CroppingDataset(test, 0, 8, 0, False)

    model = net_3d_basic.ResNet(class_labels)
    model = chainer.links.Classifier(model)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        chainer.backends.cuda.set_max_workspace_size(256 * 1024 * 1024)
        chainer.global_config.autotune = True

    optimizer = chainer.optimizers.Adam(alpha=args.learningrate, weight_decay_rate=args.weightdecay)
    optimizer.setup(model)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, repeat=True, shuffle=True)
    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.FailOnNonNumber())

    def cosine_annealing(t):
        eta_min, eta_max = (0.01, 1.0)
        warm_up_span = args.epoch // 5
        epoch = t.updater.epoch_detail
        if epoch <= warm_up_span:
            v = epoch / warm_up_span
        else:
            v = math.cos(math.pi * (epoch - warm_up_span) / (args.epoch - warm_up_span))
            v = eta_min + (eta_max - eta_min) * 0.5 * (1 + v)
        optimizer.eta = v

    trainer.extend(chainer.training.extension.make_extension((1, 'iteration'))(cosine_annealing))
    cosine_annealing(trainer)

    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.PrintReport([
        'epoch',
        'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy',
        'elapsed_time',
    ]))

    trainer.extend(chainer.training.extensions.PlotReport([
        'main/loss', 'validation/main/loss',
    ], 'epoch', file_name='loss.png'))
    trainer.extend(chainer.training.extensions.PlotReport([
        'main/accuracy', 'validation/main/accuracy',
    ], 'epoch', file_name='accuracy.png'))
    trainer.extend(
        chainer.training.extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(5, 'epoch')
    )

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
