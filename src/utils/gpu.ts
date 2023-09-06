import {ISingleValueActivation} from "mind-net.js/engine/base";
import {GPU, Kernel} from "gpu.js";

import {GpuWrappedLayer} from "../layer";
import * as FunctionUtils from "./function";

export function getActivation(layer: GpuWrappedLayer): ISingleValueActivation {
    const activation = layer.activation;
    if (!("value" in activation) || !("moment" in activation))
        throw new Error(`Activation should implements ISingleValueActivation interface`);

    return activation as ISingleValueActivation;
}

export function createForwardKernel(
    gpu: GPU, layer: GpuWrappedLayer, batchSize: number
): [Kernel, Kernel[]] {
    const activation = getActivation(layer);

    const kResult = gpu
        .addFunction(FunctionUtils.getGpuActivationFunction(activation, activation.value, "forward") as any)
        .createKernelMap({
            prime: function (input: number[][], weights: number[][], biases: number[], actualSize: number) {
                const {x: neuronIndex, y: batchIndex} = this.thread;
                if (batchIndex >= actualSize) return 0;

                let sum = biases[neuronIndex];
                // @ts-ignore
                for (let j = 0; j < this.constants.prevSize; j++) {
                    sum += input[batchIndex][j] * weights[neuronIndex][j]
                }

                return sum;
            }
        }, function (input, weights, biases, actualSize) {
            // @ts-ignore
            const primeOut = prime(input, weights, biases, actualSize);
            // @ts-ignore
            return forward(primeOut);
        })
        .setConstants({prevSize: layer.prevSize})
        .setOutput([layer.size, batchSize])
        .setTactic("precision");

    return [kResult, [kResult]];
}

export function createBackwardKernel(
    gpu: GPU, layer: GpuWrappedLayer, batchSize: number, skipErrorCalc: boolean
): [Kernel, Kernel[]] {
    const activation = getActivation(layer);

    const kGradient = gpu
        .addFunction(FunctionUtils.getGpuActivationFunction(activation, activation.moment, "backward") as any)
        .createKernel(function (prime: number[][], error: number[][], actualSize: number) {
            const {x: neuronIndex, y: batchIndex} = this.thread;
            if (batchIndex >= actualSize) return 0;
            // @ts-ignore
            return backward(prime[batchIndex][neuronIndex]) * error[batchIndex][neuronIndex];
        })
        .setOutput([layer.size, batchSize]);

    const kBiases = gpu
        .createKernel(function (gradient: number[][], actualSize: number) {
            const {x: neuronIndex} = this.thread;

            let sum = 0;
            // @ts-ignore
            for (let i = 0; i < this.constants.batchSize; i++) {
                if (i >= actualSize) break;
                sum += gradient[i][neuronIndex];
            }

            return sum;
        })
        .setConstants({batchSize: batchSize})
        .setOutput([layer.size]);

    const kWeights = gpu
        .createKernel(function (gradient: number[][], input: number[][], actualSize: number) {
            const {x: prevNeuronIndex, y: neuronIndex} = this.thread;

            let sum = 0;
            // @ts-ignore
            for (let n = 0; n < this.constants.batchSize; n++) {
                if (n >= actualSize) break;

                sum += input[n][prevNeuronIndex] * gradient[n][neuronIndex];
            }

            return sum;
        })
        .setConstants({batchSize: batchSize})
        .setOutput([layer.prevSize, layer.size]);

    let kError: any;
    if (!skipErrorCalc) {
        kError = gpu
            .createKernel(function (gradient: number[][], weights: number[][], actualSize: number) {
                const {x: prevNeuronIndex, y: batchIndex} = this.thread;
                if (batchIndex >= actualSize) return 0;

                let sum = 0;
                // @ts-ignore
                for (let neuronIndex = 0; neuronIndex < this.constants.size; neuronIndex++) {
                    sum += weights[neuronIndex][prevNeuronIndex] * gradient[batchIndex][neuronIndex];
                }

                return sum;
            })
            .setConstants({size: layer.size})
            .setOutput([layer.prevSize, batchSize]);
    }

    const kernels = [kGradient, kBiases, kWeights, kError].filter(k => !!k);
    kernels.forEach(k => k.setTactic("precision"));

    const kResult = gpu.combineKernels(
        ...kernels,
        function (error: number[][], prime: number[][], input: number[][], weights: number[][], actualSize: number) {
            const gradient = kGradient(prime, error, actualSize);
            const dB = kBiases(gradient, actualSize);
            const dW = kWeights(gradient, input, actualSize);
            const dError = !skipErrorCalc ? kError(gradient, weights, actualSize) : null;

            return {dB, dW, dError}
        });

    return [kResult, kernels];
}
