import {Matrix} from "mind-net.js";
import {GpuArray2D} from "../base";

export function splitBatches(data: Float32Array, batchSize: number): Float32Array[] {
    return Matrix.fill(
        i => data.subarray(i * batchSize, (i + 1) * batchSize),
        data.length / batchSize
    );
}

export function setArray(src: GpuArray2D, dst: Float32Array[]) {
    for (let i = 0; i < src.length; i++) {
        dst[i].set(src[i]);
    }
}