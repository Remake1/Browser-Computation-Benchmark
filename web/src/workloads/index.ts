// Workload registry. Later milestones add reduction, sort, convolution, nbody, editdistance.
import type { AnyWorkload } from '../harness/types';
import { dummy } from './dummy';
import { matmul } from './matmul';

export const WORKLOADS: AnyWorkload[] = [dummy, matmul];
