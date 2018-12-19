import chainer


class ResidualBlock3D(chainer.Chain):
    def __init__(self, channels, mid_channels, out_channels, do_pool, z_kernel):
        super().__init__()
        kernel = (z_kernel, 3, 3)
        stride = (1, 2, 2) if do_pool else (1, 1, 1)
        pad = (0, 1, 1) if z_kernel == 1 else (1, 1, 1)
        w = chainer.initializers.HeUniform()
        with self.init_scope():
            self.normalize0 = chainer.links.BatchNormalization(channels)
            self.conv1 = chainer.links.Convolution3D(channels, mid_channels, kernel, stride, pad, True, w)
            self.normalize1 = chainer.links.BatchNormalization(mid_channels)
            self.conv2 = chainer.links.Convolution3D(mid_channels, out_channels, kernel, 1, pad, True, w)
            self.normalize2 = chainer.links.BatchNormalization(out_channels)
            if do_pool or channels != out_channels:
                self.shortcut = chainer.links.Convolution3D(channels, out_channels, 1, stride, 0, True, w)
                self.normalize3 = chainer.links.BatchNormalization(out_channels)

    def __call__(self, x):
        x = self.normalize0(x)

        h = x

        h = self.conv1(h)
        h = self.normalize1(h)
        h = chainer.functions.relu(h)

        h = self.conv2(h)
        h = self.normalize2(h)

        s = x
        if h.shape != s.shape:
            s = self.shortcut(s)
            s = self.normalize3(s)

        h = h + s

        return h


def create_blocks(n, in_channels, out_channels, do_pool, z_kernel):
    sequential = chainer.Sequential()
    sequential.append(ResidualBlock3D(in_channels, out_channels, out_channels, do_pool, z_kernel))
    for _ in range(n):
        sequential.append(ResidualBlock3D(out_channels, out_channels, out_channels, False, z_kernel))
    return sequential


def z2c(x, size):
    N, C, Z, H, W = x.shape
    x = chainer.functions.reshape(x, (N, C // size, size, Z, H, W))
    x = chainer.functions.transpose(x, (0, 1, 3, 2, 4, 5))
    x = chainer.functions.reshape(x, (N, C // size, Z * size, H, W))
    return x


class ResNet(chainer.Chain):
    def __init__(self, class_labels):
        super().__init__()
        channels = 16
        w = chainer.initializers.HeUniform()
        with self.init_scope():
            self.input = chainer.links.Convolution2D(None, channels, 3, 1, 1, True, w)
            self.normalize = chainer.links.BatchNormalization(channels)

            self.layer1 = create_blocks(2, channels, channels * 2, False, 1)
            self.layer2 = create_blocks(2, channels, channels * 2, True, 3)
            self.layer3 = create_blocks(2, channels, channels * 2, True, 3)

            self.output = chainer.links.Linear(None, class_labels, True, w)
            self.output_w = chainer.Parameter(chainer.initializers.Constant(0), (1, 1))

    def __call__(self, x):
        h = x

        h = self.input(h)
        h = self.normalize(h)

        h = h[:, :, None]

        h = self.layer1(h)

        h = z2c(h, 2)

        h = self.layer2(h)

        h = z2c(h, 2)

        h = self.layer3(h)

        h = chainer.functions.max_pooling_nd(h, (1,) + h.shape[3:])
        h = self.output(h)

        output_w = chainer.functions.broadcast_to(self.output_w, h.shape)
        h = h * output_w

        return h
