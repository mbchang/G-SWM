import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
            inputs: (B*h*w, D)
            codebook: (K, D)
        """
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook ** 2, dim=1)  # (K)
            inputs_sqr = torch.sum(inputs ** 2, dim=1, keepdim=True)  # (B*h*w, 1)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs, codebook.t(), alpha=-2.0, beta=1.0)  # (B*h*w, K)

            _, indices = torch.min(distances, dim=1)  # (B*h*w)
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
            inputs: (B*h*w, D)
            codebook: (K, D)
        """
        indices = vq(inputs, codebook)  # (B*h*w)
        ctx.save_for_backward(indices, codebook)
        ctx.mark_non_differentiable(indices)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices)  # (B*h*w, D)
        codes = codes_flatten.view_as(inputs)  # (B*h*w, D)
        return (codes, indices)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
