// Workload registry. Later milestones add sort, convolution, nbody, editdistance.
import type { AnyWorkload } from '../harness/types';
import { dummy } from './dummy';
import { matmul } from './matmul';
import { reduction } from './reduction';

export const WORKLOADS: AnyWorkload[] = [dummy, matmul, reduction];
