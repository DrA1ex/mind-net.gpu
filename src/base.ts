import {Matrix1D, Matrix2D} from "mind-net.js/engine/matrix";

export type GpuArray1D = Matrix1D | Float32Array;
export type GpuArray2D = Matrix2D | Float32Array[] | GpuArray1D[];