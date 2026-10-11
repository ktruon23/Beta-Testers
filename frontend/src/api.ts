// Types mirror backend/shared/schemas.py. The frontend only talks to the orchestrator.

export const API = import.meta.env.VITE_API_URL ?? "/api";

export type Level = "beginner" | "intermediate" | "expert";

export const GRIPS = ["jug", "crimp", "sloper", "pinch", "pocket"] as const;
export type Grip = (typeof GRIPS)[number];

export interface Hold {
  id: number;
  x: number;
  y: number;
  type: string;
  color?: string | null;
}

export interface Climber {
  height_in: number;
  wingspan_in: number;
  level: Level;
}

export interface Move {
  from_id: number;
  to_id: number;
  distance_in: number;
  reach_ratio: number;
  dx_in: number;
  dy_in: number;
  from_type: string;
  to_type: string;
}

export interface PlanAlternative {
  strategy: string;
  path: number[];
  total_cost: number;
  max_reach_ratio: number;
  move_count: number;
  worst_hold_type: string;
  ok: boolean;
  error?: string | null;
}

export interface PlanResponse {
  ok: boolean;
  path: number[];
  moves: Move[];
  reach_in: number;
  strategy: string;
  reasoning: string;
  alternatives: PlanAlternative[];
  error?: string | null;
  source: string;
}

export interface RouteDifficulty {
  grade: string;
  score: number;
  label: string;
  reasoning: string;
  assumed_holds: number[];
  default_hold_type: string;
}

export interface CoachResponse {
  summary: string;
  steps: { move: number; instruction: string }[];
  tips: string[];
  source: string;
}

/** What to sketch on the photo for one message (image pixels). Sent by the orchestrator in payload.draw. */
export interface Sketch {
  path: [number, number][];
  highlights: { from: [number, number]; to: [number, number]; label: string }[];
  marks: { at: [number, number]; label: string }[];
  crossed: [number, number][];
}

export interface TraceEntry {
  step: number;
  sender: string;
  receiver: string;
  summary: string;
  payload: Record<string, unknown>;
}

export interface SolveResponse {
  ok: boolean;
  status: string;
  holds: Hold[];
  start_id?: number | null;
  finish_id?: number | null;
  plan?: PlanResponse | null;
  critic_approved: boolean;
  difficulty?: RouteDifficulty | null;
  beta?: CoachResponse | null;
  trace: TraceEntry[];
  rounds: number;
  orchestrator_source: string;
}

export interface DetectResponse {
  holds: Hold[];
  image_width: number;
  image_height: number;
  inches_per_pixel?: number | null;
  scale_method?: string | null;
  colors: Record<string, number>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Can't reach the orchestrator. Is run_agents.sh running?");
  }
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    const detail = data.detail;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors: [{loc, msg}, ...]
    if (Array.isArray(detail)) return detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join("; ");
  } catch {
    /* not JSON */
  }
  if (res.status === 502 || res.status === 504) return "An agent didn't answer. Check that all five services are running.";
  return `The orchestrator returned ${res.status}.`;
}

export function detect(image_b64: string) {
  return post<DetectResponse>("/detect", { image_b64 });
}

export function solvePhoto(req: {
  image_b64: string;
  climber: Climber;
  route_color?: string | null;
}) {
  return post<SolveResponse>("/solve_photo", req);
}

/** Like solvePhoto, but calls onEntry with each agent message as it happens. */
export type Point = [number, number];

export async function solvePhotoStream(
  req: {
    image_b64: string;
    climber: Climber;
    route_color?: string | null;
    start_point?: Point | null; // tapped on the photo; the backend snaps it to the nearest hold
    finish_point?: Point | null;
    grips?: { x: number; y: number; type: Grip }[]; // grips the climber set on holds
  },
  onEntry: (entry: TraceEntry) => void,
): Promise<SolveResponse> {
  let res: Response;
  try {
    res = await fetch(`${API}/solve_photo/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    throw new Error("Can't reach the orchestrator. Is run_agents.sh running?");
  }
  if (!res.ok || !res.body) throw new Error(await errorText(res));

  // Newline-delimited JSON: {type: "trace" | "result" | "error", ...}
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "trace") onEntry(event.entry);
      else if (event.type === "result") return event.result;
      else if (event.type === "error") throw new Error(event.detail);
    }
  }
  throw new Error("The orchestrator stopped before the route was finished.");
}

export async function healthAll(): Promise<Record<string, { status?: string }>> {
  const res = await fetch(`${API}/health/all`);
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}

/** Shrink big phone photos before upload. Hold coordinates come back in this image's pixels. */
export async function loadPhoto(file: File, maxSide = 1600): Promise<{ dataUrl: string; b64: string; width: number; height: number }> {
  const bitmap = await createImageBitmap(file); // applies EXIF rotation in current browsers
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
  return { dataUrl, b64: dataUrl.split(",")[1], width, height };
}

/** "5 ft 8 in", "5'8", "68", "68 in", "173 cm" -> inches. */
export function parseLength(text: string): number | null {
  const t = text.trim().toLowerCase();
  if (!t) return null;
  const cm = t.match(/^(\d+(?:\.\d+)?)\s*cm$/);
  if (cm) return +cm[1] / 2.54;
  const ftIn = t.match(/^(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')\s*(?:(\d+(?:\.\d+)?)\s*(?:in|inches|"|'')?)?$/);
  if (ftIn) return +ftIn[1] * 12 + (ftIn[2] ? +ftIn[2] : 0);
  const inches = t.match(/^(\d+(?:\.\d+)?)\s*(?:in|inches|")?$/);
  if (inches) return +inches[1];
  return null;
}

export function formatInches(n: number): string {
  return `${Math.round(n)} in`;
}
