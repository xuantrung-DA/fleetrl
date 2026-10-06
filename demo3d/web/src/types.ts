export type Cell = [number, number];

export interface PortMap {
  id: number;
  station: number;
  cell: Cell;
  power_w: number;
  compatible: string[];
}

export interface MapData {
  name: string;
  width: number;
  height: number;
  walls: Cell[];
  pickups: Cell[];
  dropoffs: Cell[];
  parking: Cell[];
  ports: PortMap[];
  queue_cells: Record<string, Cell[]>;
}

export interface RobotFrame {
  id: number;
  type: "L" | "M" | "H";
  cell: Cell;
  status: string;
  soc: number;
  load_kg: number;
  task_id: number | null;
  move_to?: Cell | null;
  move_remaining_s?: number;
  edge_duration_s?: number;
  route?: Cell[];
  battery_wh: number;
  capacity_wh: number;
}

export interface PortFrame {
  id: number;
  occupied_by: number | null;
  charging_robot: number | null;
  queue_robot_ids: number[];
}

export interface DecisionFrame {
  profile: number | null;
  solver_status: string;
  fallback_reason: string | null;
  total_ms: number;
  time_s: number | null;
}

export interface Frame {
  type: "frame";
  t: number;
  observed_at: number;
  robots: RobotFrame[];
  ports: PortFrame[];
  blocked_cells: Cell[];
  kpi: Record<string, unknown>;
  decision: DecisionFrame | null;
  modified: boolean;
}

export interface MethodAvailability {
  name: string;
  available: boolean;
  reason: string | null;
}

export interface SessionStatus {
  id: string;
  method: string;
  scenario: string;
  seed: number;
  status: string;
  paused: boolean;
  speed: number;
  modified: boolean;
  error: string | null;
  replay: string;
  time_s: number;
}

export interface Bootstrap {
  methods: Record<string, MethodAvailability>;
  scenarios: string[];
  default_scenario: string;
  default_seed: number;
  session: SessionStatus | null;
}

export interface ReplayHeader {
  type: "header";
  map: MapData;
  session: SessionStatus;
}
