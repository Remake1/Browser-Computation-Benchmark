// Workload registry. Milestone 4 adds editdistance.
import type { AnyWorkload } from '../harness/types';
import { convolution } from './convolution';
import { dummy } from './dummy';
import { matmul } from './matmul';
import { nbody } from './nbody';
import { reduction } from './reduction';
import { sort } from './sort';

export const WORKLOADS: AnyWorkload[] = [dummy, matmul, reduction, sort, convolution, nbody];
