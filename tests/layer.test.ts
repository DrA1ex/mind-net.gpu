import {IActivation, ISingleValueActivation} from "mind-net.js/engine/base";
import {Matrix1D} from "mind-net.js/engine/matrix";
import {Dense, SgdOptimizer, Activations, Matrix} from "mind-net.js";
import {GPU} from "gpu.js"

import {GpuWrappedLayer} from "../src/layer";
import * as ArrayUtils from "./utils/array";

const gpu = new GPU({mode: "cpu"});

function _wrappedLayer(size = 8, prevSize = 4) {
    const layer = new Dense(size);
    layer.build(2, prevSize, false);
    return new GpuWrappedLayer(gpu, layer, 2, 1);
}

class CustomActivation implements IActivation, ISingleValueActivation {
    value(x: number): number {
        return x * x / 2;
    }

    moment(x: number): number {
        return 2 * x;
    }

    forward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return Matrix.matrix1d_unary_op(input, this.value.bind(this), dst);
    }

    backward(input: Matrix1D, dst?: Matrix1D): Matrix1D {
        return Matrix.matrix1d_unary_op(input, this.moment.bind(this), dst);
    }
}

describe.each([1, 4, 8])
("Layer size: %d", size => {
    describe.each([
        Activations.SigmoidActivation,
        Activations.ReluActivation,
        Activations.LeakyReluActivation,
        Activations.TanhActivation,
        Activations.LinearActivation,
        CustomActivation
    ])
    ("%p", activation => {
        // Due to the usage of Float32 in GPU and Float64 in JS, we need a bigger epsilon.
        const eps = 1e-3;

        test("Layer.forward", () => {
            const layer = new Dense(size, {activation: activation as any});
            layer.build(1, 4, false);
            const wrapped = new GpuWrappedLayer(gpu, layer, 1, 1);

            const input = Matrix.random_1d(32);

            const layerOut = layer.step(input);
            const wrapperOut = wrapped.forward([input], 1, false);

            ArrayUtils.arrayCloseTo(layerOut, wrapped.prime[0], eps);

            const layerActivationOut = layer.activation.forward(layerOut);
            ArrayUtils.arrayCloseTo(layerActivationOut, wrapperOut[0], eps);
        });

        test("Layer.backward", () => {
            const optimizer = new SgdOptimizer();
            const layer = new Dense(size, {activation: activation as any});
            layer.build(1, 4, false);
            const wrapped = new GpuWrappedLayer(gpu, layer, 2, 1);

            const forwardInput = Matrix.random_1d(8);
            const error = Matrix.random_1d(size);

            const prime = layer.step(forwardInput);
            layer.activation.forward(prime, layer.activationOutput);

            const dWeights = Matrix.zero_2d(layer.size, layer.prevSize)
            const dBiases = Matrix.zero(layer.size);

            const gradient = optimizer.step(layer, error, 0);
            const dError = layer.backward(gradient, dWeights, dBiases);

            wrapped.forward([forwardInput], 1, false);
            const res = wrapped.backward([error], 1);

            //ArrayUtils.arrayCloseTo_2d(dWeights, res.dW, eps);
            //ArrayUtils.arrayCloseTo(dBiases, res.dB, eps);
            ArrayUtils.arrayCloseTo(dError, res.dError[0], eps);
        });
    })
});

describe("GpuWrappedLayer.destroy", () => {
    test("Should successfully free GPU resources", () => {
        const wrapped = _wrappedLayer();
        wrapped.destroy();

        expect(wrapped.forwardKernel).toBeUndefined();
        expect(wrapped.backwardKernel).toBeUndefined();
    });

    test("Should not fail with continuous destroy calls", () => {
        const wrapped = _wrappedLayer();

        wrapped.destroy();
        wrapped.destroy();
    })

    describe("Should fail when used after terminate", () => {
        test.failing("forward", () => {
            const wrapped = _wrappedLayer();
            wrapped.destroy();

            wrapped.forward([new Array(8)], 1, false);
        });

        test.failing("backward", () => {
            const wrapped = _wrappedLayer();
            wrapped.destroy();

            wrapped.backward([new Array(4)], 1);
        });
    });
});