// Workload registry. Milestone 4 adds sort and editdistance.
import type { AnyWorkload } from '../harness/types';
import { convolution } from './convolution';
import { dummy } from './dummy';
import { matmul } from './matmul';
import { nbody } from './nbody';
import { reduction } from './reduction';

export const WORKLOADS: AnyWorkload[] = [dummy, matmul, reduction, convolution, nbody];
