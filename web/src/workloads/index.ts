// Workload registry. Later milestones add matmul, reduction, sort, convolution, nbody, editdistance.
import type { AnyWorkload } from '../harness/types';
import { dummy } from './dummy';

export const WORKLOADS: AnyWorkload[] = [dummy];
