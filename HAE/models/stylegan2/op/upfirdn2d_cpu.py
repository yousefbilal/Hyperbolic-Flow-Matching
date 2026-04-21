import os
import torch
import torch.nn.functional as F
from torch.autograd import Function

module_path = os.path.dirname(__file__)

# Try to load compiled CUDA/C++ extension, else fallback to CPU
try:
    from torch.utils.cpp_extension import load
    upfirdn2d_op = load(
        'upfirdn2d',
        sources=[
            os.path.join(module_path, 'upfirdn2d.cpp'),
            os.path.join(module_path, 'upfirdn2d_kernel.cu'),
        ],
    )
    _has_extension = True
except Exception as e:
    print(f"UpFirDn2d extension unavailable, using CPU fallback. ({e})")
    upfirdn2d_op = None
    _has_extension = False


# =========================
# GPU / compiled version
# =========================
if _has_extension:

    class UpFirDn2d(Function):
        @staticmethod
        def forward(ctx, input, kernel, up, down, pad):
            up_x, up_y = up
            down_x, down_y = down
            pad_x0, pad_x1, pad_y0, pad_y1 = pad

            kernel_h, kernel_w = kernel.shape
            batch, channel, in_h, in_w = input.shape
            ctx.in_size = input.shape

            input = input.reshape(-1, in_h, in_w, 1)
            ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

            out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
            out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
            ctx.out_size = (out_h, out_w)
            ctx.up = (up_x, up_y)
            ctx.down = (down_x, down_y)
            ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

            out = upfirdn2d_op.upfirdn2d(
                input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
            )
            out = out.view(-1, channel, out_h, out_w)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            kernel, grad_kernel = ctx.saved_tensors
            grad_input = upfirdn2d_op.upfirdn2d(
                grad_output.reshape(-1, ctx.out_size[0], ctx.out_size[1], 1),
                grad_kernel,
                ctx.down[0], ctx.down[1],
                ctx.up[0], ctx.up[1],
                ctx.pad[0], ctx.pad[1],
                ctx.pad[2], ctx.pad[3],
            )
            grad_input = grad_input.view(ctx.in_size)
            return grad_input, None, None, None, None

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        return UpFirDn2d.apply(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )


# =========================
# CPU fallback
# =========================
else:

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
        """Simplified CPU version of upfirdn2d."""
        if isinstance(pad, int):
            pad = (pad, pad)
        if isinstance(up, int):
            up = (up, up)
        if isinstance(down, int):
            down = (down, down)

        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1 = pad
        pad_y0, pad_y1 = pad

        # Upsample
        if up_x > 1 or up_y > 1:
            input = F.interpolate(input, scale_factor=up_x, mode='nearest')

        # Pad
        if pad_x0 > 0 or pad_x1 > 0 or pad_y0 > 0 or pad_y1 > 0:
            input = F.pad(input, (pad_x0, pad_x1, pad_y0, pad_y1))

        # Convolve with kernel
        kernel = kernel.to(input.device, dtype=input.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(input.shape[1], 1, 1, 1)  # [C, 1, kH, kW]

        input = F.conv2d(input, kernel, groups=input.shape[1])




        # Downsample
        if down_x > 1 or down_y > 1:
            input = input[:, :, ::down_y, ::down_x]

        return input
