import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  GRIPS,
  detect,
  formatInches,
  healthAll,
  loadPhoto,
  parseLength,
  solvePhotoStream,
  type DetectResponse,
  type Grip,
  type Hold,
  type Level,
  type Point,
  type Sketch,
  type SolveResponse,
  type TraceEntry,
} from "./api";

type Photo = { dataUrl: string; b64: string; width: number; height: number };

type Picks = { start: Point | null; finish: Point | null };
type Grips = Record<string, Grip>; // keyed by holdKey()

const holdKey = (h: { x: number; y: number }) => `${h.x},${h.y}`;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
/** How long a chat message stays as the newest one: long enough to read it. */
const readingTime = (entry: TraceEntry) => Math.min(3200, 1100 + entry.summary.length * 20);
type ProfileArgs = { height: number; wingspan: number; level: Level; color: string };

type Attempt = {
  name: string;
  path: number[];
  rejected: number[]; // holds the Critic asked the Planner to avoid after this try
  traceEnd: number; // how much of the conversation had happened by this try
  note: string;
};

const AGENTS: Record<string, { short: string; className?: string }> = {
  Vision: { short: "V" },
  Planner: { short: "P" },
  Critic: { short: "C" },
  Coach: { short: "Co", className: "coach-avatar" },
  Orchestrator: { short: "O", className: "orchestrator-avatar" },
};

/* ---------------- helpers ---------------- */

/** Split the trace into the Planner's tries, plus the final route. */
function buildAttempts(result: SolveResponse): Attempt[] {
  const trace = result.trace;
  const tries: Attempt[] = [];
  trace.forEach((entry, i) => {
    const path = entry.payload?.path;
    if (entry.sender === "Planner" && Array.isArray(path)) {
      const verdict = trace.slice(i + 1).find((e) => e.sender === "Critic");
      const rejected = verdict && Array.isArray(verdict.payload?.avoid_holds) ? (verdict.payload.avoid_holds as number[]) : [];
      const end = verdict ? trace.indexOf(verdict) + 1 : i + 1;
      const approved = verdict?.payload?.approved === true;
      tries.push({
        name: `Try ${tries.length + 1}`,
        path: path as number[],
        rejected,
        traceEnd: end,
        note: approved ? "Critic approved" : verdict ? "Critic pushed back" : "Proposed",
      });
    }
  });

  const finalPath = result.plan?.path ?? [];
  const allRejected = [...new Set(tries.flatMap((t) => t.rejected))].filter((id) => !finalPath.includes(id));
  const final: Attempt = {
    name: "Final",
    path: finalPath,
    rejected: allRejected,
    traceEnd: trace.length,
    note: result.critic_approved ? "Ready to climb" : "Best route found",
  };
  // The last try is usually the final route; don't show it twice.
  const last = tries.at(-1);
  if (last && last.path.join() === finalPath.join()) tries.pop();
  return [...tries, final];
}

function holdLabel(type: string | undefined) {
  return type && type !== "unknown" ? type : "hold";
}

/** Fit a w×h image inside its container (like object-fit: contain) and report the box it lands in. */
function useContainBox(w: number, h: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ left: 0, top: 0, width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !w || !h) return;
    const update = () => {
      const cw = el.clientWidth;
      const ch = el.clientHeight;
      const s = Math.min(cw / w, ch / h);
      setBox({ width: w * s, height: h * s, left: (cw - w * s) / 2, top: (ch - h * s) / 2 });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [w, h]);
  return { ref, box };
}

function WallStage({ photo, className, children }: { photo: Photo; className: string; children?: ReactNode }) {
  const { ref, box } = useContainBox(photo.width, photo.height);
  return (
    <div className={className} ref={ref}>
      <div className="wall-stage" style={box}>
        <img src={photo.dataUrl} alt="Your climbing wall" />
        {children}
      </div>
    </div>
  );
}

const pct = (v: number, of: number) => `${(v / of) * 100}%`;

function sketchOf(entry: TraceEntry | undefined): Sketch | null {
  const draw = entry?.payload?.draw as Sketch | undefined;
  return draw && (draw.path.length || draw.highlights.length || draw.marks.length || draw.crossed.length) ? draw : null;
}

/** Draws what an agent is talking about: the route it proposes, the move it calls out, the holds it skips. */
function SketchLayer({ sketch, photo }: { sketch: Sketch; photo: Photo }) {
  const at = ([x, y]: [number, number]) => ({ left: pct(x, photo.width), top: pct(y, photo.height) });
  const pathTime = 0.25 * Math.max(1, sketch.path.length - 1); // seconds to draw the route, hold by hold
  return (
    <div className="sketch-layer">
      <svg viewBox={`0 0 ${photo.width} ${photo.height}`} preserveAspectRatio="none">
        {sketch.path.length > 1 && (
          <polyline
            className="sketch-path"
            pathLength={1}
            style={{ animationDuration: `${pathTime}s` }}
            points={sketch.path.map(([x, y]) => `${x},${y}`).join(" ")}
          />
        )}
        {sketch.highlights.map((h, i) => (
          <line
            key={i}
            className={`sketch-highlight ${h.label === "skip" || h.label === "too far" ? "bad" : ""}`}
            pathLength={1}
            style={{ animationDelay: `${sketch.path.length > 1 ? pathTime : 0}s` }}
            x1={h.from[0]}
            y1={h.from[1]}
            x2={h.to[0]}
            y2={h.to[1]}
          />
        ))}
      </svg>
      {sketch.path.map((p, i) => (
        <span key={`p${i}`} className="sketch-dot" style={{ ...at(p), animationDelay: `${i * 0.25}s` }}>
          {i === 0 ? "S" : i}
        </span>
      ))}
      {sketch.crossed.map((p, i) => (
        <span key={`x${i}`} className="crossed-hold sketch-crossed" style={at(p)}>
          <i />
          <i />
        </span>
      ))}
      {sketch.marks.map((m, i) => (
        <span key={`m${i}`} className={`sketch-mark ${m.label}`} style={at(m.at)}>
          {m.label}
        </span>
      ))}
      {sketch.highlights.map((h, i) => (
        <span
          key={`l${i}`}
          className={`sketch-label ${h.label === "skip" || h.label === "too far" ? "bad" : ""}`}
          style={{
            ...at([(h.from[0] + h.to[0]) / 2, (h.from[1] + h.to[1]) / 2]),
            animationDelay: `${sketch.path.length > 1 ? pathTime + 0.3 : 0.3}s`,
          }}
        >
          {h.label}
        </span>
      ))}
    </div>
  );
}

/* ---------------- page pieces ---------------- */

function Header() {
  return (
    <header className="site-header">
      <a className="wordmark" href="#top" aria-label="Beta Agents home">
        <span className="wordmark-mark" aria-hidden="true">
          β
        </span>
        Beta Agents
      </a>
      <nav aria-label="Page sections">
        <a href="#wall">Your wall</a>
        <a href="#route">Your route</a>
        <a href="#beta">How to climb it</a>
      </nav>
    </header>
  );
}

function HealthBanner() {
  const [down, setDown] = useState<string[] | null>(null);
  useEffect(() => {
    healthAll()
      .then((h) => setDown(Object.entries(h).filter(([, v]) => v?.status !== "ok").map(([k]) => k)))
      .catch(() => setDown(["orchestrator"]));
  }, []);
  if (!down || down.length === 0) return null;
  return (
    <p className="health-banner" role="status">
      {down.includes("orchestrator")
        ? "Can't reach the orchestrator on port 8000. Start the agents with run_agents.sh, then refresh."
        : `These agents aren't answering: ${down.join(", ")}. Routes may come back without them.`}
    </p>
  );
}

function WallUpload({
  photo,
  detection,
  detecting,
  color,
  picks,
  picking,
  onPick,
  grips,
  setGrip,
  onPhoto,
}: {
  grips: Grips;
  setGrip: (hold: Hold, grip: Grip | null) => void;
  photo: Photo | null;
  detection: DetectResponse | null;
  detecting: boolean;
  color: string;
  picks: Picks;
  picking: "start" | "finish" | null;
  onPick: (hold: Hold) => void;
  onPhoto: (file: File) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const choose = () => fileRef.current?.click();
  const [editing, setEditing] = useState<Hold | null>(null);
  useEffect(() => setEditing(null), [photo, detection, picking, color]);
  useEffect(() => {
    if (!editing) return;
    const close = (e: KeyboardEvent) => e.key === "Escape" && setEditing(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [editing]);

  return (
    <div
      className={`wall-upload ${dragging ? "dragging" : ""} ${photo ? "has-photo" : ""} ${picking ? "picking" : ""}`}
      // Before there's a photo the whole area opens the file picker; after, only "Change photo" does.
      role={photo ? undefined : "button"}
      tabIndex={photo ? undefined : 0}
      onClick={photo ? undefined : choose}
      onKeyDown={photo ? undefined : (e) => (e.key === "Enter" || e.key === " ") && choose()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file?.type.startsWith("image/")) onPhoto(file);
      }}
    >
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onPhoto(file);
          e.target.value = "";
        }}
      />
      {photo ? (
        <>
          <WallStage photo={photo} className="upload-stage">
            {detection?.holds
              .filter((h) => !color || h.color === color)
              .map((h, i) =>
                picking ? (
                  <button
                    key={h.id}
                    type="button"
                    className="pick-hold"
                    aria-label={`Use this ${holdLabel(h.type)} as the ${picking}`}
                    style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }}
                    onClick={() => onPick(h)}
                  />
                ) : (
                  <button
                    key={h.id}
                    type="button"
                    className={`detected-hold grip-target ${grips[holdKey(h)] ? "has-grip" : ""} ${editing?.id === h.id ? "editing" : ""}`}
                    aria-label={`Set the grip for this hold${grips[holdKey(h)] ? ` (now ${grips[holdKey(h)]})` : ""}`}
                    style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height), animationDelay: `${Math.min(i * 12, 500)}ms` }}
                    onClick={() => setEditing(editing?.id === h.id ? null : h)}
                  >
                    {grips[holdKey(h)] && <small>{grips[holdKey(h)]}</small>}
                  </button>
                ),
              )}
            {editing && (
              <div
                className={`grip-menu ${editing.x > photo.width * 0.65 ? "flip-x" : ""} ${editing.y > photo.height * 0.6 ? "flip-y" : ""}`}
                role="dialog"
                aria-label="Choose a grip"
                style={{ left: pct(editing.x, photo.width), top: pct(editing.y, photo.height) }}
              >
                <strong>What grip is this?</strong>
                <div>
                  {GRIPS.map((g) => (
                    <button
                      key={g}
                      type="button"
                      className={grips[holdKey(editing)] === g ? "active" : ""}
                      onClick={() => {
                        setGrip(editing, g);
                        setEditing(null);
                      }}
                    >
                      {g}
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  className="grip-clear"
                  onClick={() => {
                    setGrip(editing, null);
                    setEditing(null);
                  }}
                >
                  {grips[holdKey(editing)] ? "Clear" : "Not sure"}
                </button>
              </div>
            )}
            {(["start", "finish"] as const).map((which) => {
              const p = picks[which];
              return p ? (
                <span key={which} className={`sketch-mark ${which}`} style={{ left: pct(p[0], photo.width), top: pct(p[1], photo.height) }}>
                  {which}
                </span>
              ) : null;
            })}
            {detecting && (
              <div className="detect-loading" aria-hidden="true">
                <span className="scan-line" />
              </div>
            )}
          </WallStage>
          <span className="detect-status" role="status">
            {detecting ? (
              <>
                <span className="detect-spinner" aria-hidden="true" />
                Looking for holds…
              </>
            ) : picking ? (
              `Tap a hold to use as your ${picking}`
            ) : detection ? (
              `${color ? `${detection.colors[color] ?? 0} ${color}` : `Vision found ${detection.holds.length}`} holds · tap one to set its grip`
            ) : (
              "Photo ready"
            )}
          </span>
          <button type="button" className="replace-photo" onClick={choose}>
            Change photo
          </button>
        </>
      ) : (
        <div className="upload-prompt">
          <span className="upload-symbol" aria-hidden="true">
            +
          </span>
          <strong>Add a photo of your wall</strong>
          <span>Drop it here or choose a file</span>
          <small>Try to get the whole wall in frame, straight on.</small>
        </div>
      )}
    </div>
  );
}

function ProfileForm({
  colors,
  color,
  setColor,
  canSubmit,
  solving,
  error,
  onFind,
  pickBar,
}: {
  colors: Record<string, number>;
  color: string;
  setColor: (color: string) => void;
  pickBar: ReactNode;
  canSubmit: boolean;
  solving: boolean;
  error: string | null;
  onFind: (args: { height: number; wingspan: number; level: Level; color: string }) => void;
}) {
  const [height, setHeight] = useState("5 ft 8 in");
  const [wingspan, setWingspan] = useState("70 in");
  const [level, setLevel] = useState<Level>("intermediate");
  const [formError, setFormError] = useState<string | null>(null);

  return (
    <form
      className="profile-form"
      onSubmit={(event) => {
        event.preventDefault();
        const h = parseLength(height);
        const w = parseLength(wingspan);
        if (!h || h < 36 || h > 96) return setFormError("Enter your height like 5 ft 8 in, 68 in or 173 cm.");
        if (!w || w < 36 || w > 110) return setFormError("Enter your wingspan like 70 in or 178 cm.");
        setFormError(null);
        onFind({ height: h, wingspan: w, level, color });
      }}
    >
      <h2>About you</h2>
      <p>We use your measurements to rule out moves that are too far.</p>
      <label>
        <span>Height</span>
        <input value={height} onChange={(e) => setHeight(e.target.value)} aria-label="Height" />
      </label>
      <label>
        <span>Wingspan</span>
        <input value={wingspan} onChange={(e) => setWingspan(e.target.value)} aria-label="Wingspan" />
      </label>
      <label>
        <span>Climbing level</span>
        <select value={level} onChange={(e) => setLevel(e.target.value as Level)}>
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="expert">Expert</option>
        </select>
      </label>
      <label>
        <span>Route color</span>
        <select value={color} onChange={(e) => setColor(e.target.value)}>
          <option value="">Any hold</option>
          {Object.entries(colors).map(([name, n]) => (
            <option key={name} value={name}>
              {name[0].toUpperCase() + name.slice(1)} ({n} holds)
            </option>
          ))}
        </select>
      </label>
      <div className="form-field">
        <span>Start and finish</span>
        {pickBar}
      </div>
      {(formError || error) && (
        <p className="form-error" role="alert">
          {formError || error}
        </p>
      )}
      <button className="orange-button" type="submit" disabled={!canSubmit || solving}>
        {solving ? "The agents are working…" : canSubmit ? "Find a route" : "Add a photo first"}
      </button>
      {solving && <small className="form-hint">The Planner and Critic can take a minute to agree.</small>}
    </form>
  );
}

function RouteWall({
  photo,
  holds,
  color,
  attempt,
  moveTypes,
  activeHold,
  setActiveHold,
  picking,
  onPick,
  sketch,
}: {
  photo: Photo;
  holds: Hold[];
  color: string;
  sketch: { key: number; sketch: Sketch } | null;
  picking: "start" | "finish" | null;
  onPick: (hold: Hold) => void;
  attempt: Attempt;
  moveTypes: string[];
  activeHold: number;
  setActiveHold: (index: number) => void;
}) {
  const byId = useMemo(() => new Map(holds.map((h) => [h.id, h])), [holds]);
  const points = attempt.path.map((id) => byId.get(id)).filter((h): h is Hold => !!h);
  const line = points.map((h) => `${h.x},${h.y}`).join(" ");

  return (
    <WallStage photo={photo} className={`route-wall ${picking ? "picking" : ""} ${sketch ? "sketching" : ""}`}>
      {sketch && <SketchLayer key={sketch.key} sketch={sketch.sketch} photo={photo} />}
      {holds
        .filter((h) => !color || h.color === color)
        .map((h) =>
          picking ? (
            <button
              key={h.id}
              className="pick-hold"
              aria-label={`Use this ${holdLabel(h.type)} as the ${picking}`}
              style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }}
              onClick={() => onPick(h)}
            />
          ) : (
            <span key={h.id} className="detected-hold faint" style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }} />
          ),
        )}
      <svg className="drawn-route" viewBox={`0 0 ${photo.width} ${photo.height}`} preserveAspectRatio="none">
        <polyline points={line} />
      </svg>
      {attempt.rejected.map((id) => {
        const h = byId.get(id);
        return h ? (
          <span
            key={`x${id}`}
            className="crossed-hold"
            title="The Critic ruled this hold out"
            style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }}
          >
            <i />
            <i />
          </span>
        ) : null;
      })}
      {points.map((h, index) => (
        <button
          key={`${h.id}-${index}`}
          aria-label={index === 0 ? "Start hold" : `Move ${index}: ${holdLabel(moveTypes[index - 1])}`}
          className={`route-point ${activeHold === index ? "active" : ""}`}
          style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }}
          onClick={() => setActiveHold(index)}
        >
          <span>{index === 0 ? "S" : index}</span>
          <small>
            {index === 0 ? "start" : index === points.length - 1 ? "finish" : holdLabel(moveTypes[index - 1] ?? h.type)}
          </small>
        </button>
      ))}
    </WallStage>
  );
}

/** Who the last message was handed to, i.e. who is working right now. */
function workingAgent(trace: TraceEntry[]) {
  const last = trace.at(-1);
  if (!last) return "Vision";
  return last.receiver !== last.sender ? last.receiver : "Orchestrator";
}

function Conversation({
  trace,
  upTo,
  footer,
  live = false,
  drawnStep,
  onHover,
}: {
  trace: TraceEntry[];
  upTo: number;
  footer: string;
  live?: boolean;
  drawnStep?: number;
  onHover: (entry: TraceEntry | null) => void;
}) {
  const shown = trace.slice(0, upTo);
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (live && el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [live, shown.length]);

  return (
    <aside className="conversation" aria-live={live ? "polite" : undefined}>
      <div className="conversation-heading">
        <h2>The route discussion</h2>
        <p>
          {live
            ? "Watch the agents work it out on your wall."
            : "Hover over a message to see it on the wall."}
        </p>
      </div>
      <div className="conversation-scroll" ref={scrollRef}>
        {shown.map((entry) => {
          const agent = AGENTS[entry.sender] ?? { short: entry.sender.slice(0, 2) };
          const drawable = !!sketchOf(entry);
          return (
            <div
              className={`message ${drawable ? "drawable" : ""} ${drawnStep === entry.step ? "drawn" : ""}`}
              key={entry.step}
              tabIndex={drawable ? 0 : undefined}
              onMouseEnter={drawable ? () => onHover(entry) : undefined}
              onMouseLeave={drawable ? () => onHover(null) : undefined}
              onFocus={drawable ? () => onHover(entry) : undefined}
              onBlur={drawable ? () => onHover(null) : undefined}
            >
              <span className={`agent-avatar ${agent.className ?? ""}`}>{agent.short}</span>
              <div>
                <b>
                  {entry.sender}
                  {entry.receiver !== entry.sender && <span className="to-agent"> to {entry.receiver}</span>}
                </b>
                <p>{entry.summary}</p>
              </div>
            </div>
          );
        })}
        {live && (
          <div className="message typing">
            <span className={`agent-avatar ${AGENTS[workingAgent(shown)]?.className ?? ""}`}>
              {AGENTS[workingAgent(shown)]?.short ?? "?"}
            </span>
            <div>
              <b>{workingAgent(shown)}</b>
              <p>
                is working<span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
              </p>
            </div>
          </div>
        )}
      </div>
      {footer && <p className="conversation-note">{footer}</p>}
    </aside>
  );
}

function RoundTabs({
  attempts,
  round,
  setRound,
}: {
  attempts: Attempt[];
  round: number;
  setRound: (round: number) => void;
}) {
  return (
    <div className="round-tabs" aria-label="Route attempts">
      {attempts.map((a, index) => (
        <button key={a.name} className={round === index ? "active" : ""} onClick={() => setRound(index)}>
          {a.name}
        </button>
      ))}
      <span>{attempts[round]?.note}</span>
    </div>
  );
}

function PickBar({
  picks,
  picking,
  setPicking,
  onAuto,
  before = false,
  disabled = false,
}: {
  picks: Picks;
  picking: "start" | "finish" | null;
  setPicking: (mode: "start" | "finish" | null) => void;
  onAuto: () => void;
  before?: boolean; // shown in the form, before a route exists
  disabled?: boolean;
}) {
  if (picking) {
    return (
      <div className="pick-bar picking" role="status">
        <span>Tap a hold on the photo to use as your {picking}.</span>
        <button type="button" onClick={() => setPicking(null)}>
          Cancel
        </button>
      </div>
    );
  }
  const mine = [picks.start && "start", picks.finish && "finish"].filter(Boolean).join(" and ");
  return (
    <div className={`pick-bar ${before ? "in-form" : ""}`}>
      <span>
        {mine
          ? `You picked the ${mine}.`
          : before
            ? "The agents will choose, or pick your own."
            : "The agents picked the start and finish."}
      </span>
      <button type="button" disabled={disabled} onClick={() => setPicking("start")}>
        Pick the start
      </button>
      <button type="button" disabled={disabled} onClick={() => setPicking("finish")}>
        Pick the finish
      </button>
      {mine && (
        <button type="button" onClick={onAuto}>
          Let the agents choose
        </button>
      )}
    </div>
  );
}

function MoveList({
  result,
  activeHold,
  setActiveHold,
}: {
  result: SolveResponse;
  activeHold: number;
  setActiveHold: (index: number) => void;
}) {
  const moves = result.plan?.moves ?? [];
  const steps = new Map((result.beta?.steps ?? []).map((s) => [s.move, s.instruction]));
  const reach = result.plan?.reach_in ?? 0;

  return (
    <ol className="move-list">
      {moves.map((move, index) => (
        <li key={`${move.from_id}-${move.to_id}`}>
          <button
            className={activeHold === index + 1 ? "active" : ""}
            onClick={() => {
              setActiveHold(index + 1);
              document.getElementById("route-image")?.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
          >
            <span className="move-index">{index + 1}</span>
            <span className="move-copy">
              <strong>
                To the {holdLabel(move.to_type)}, {formatInches(move.distance_in)} away.
              </strong>{" "}
              {steps.get(index + 1) ?? ""}
            </span>
            <small title={`${formatInches(move.distance_in)} of your ${formatInches(reach)} reach`}>
              {Math.round(move.reach_ratio * 100)}%
            </small>
          </button>
        </li>
      ))}
    </ol>
  );
}

function Alternatives({
  result,
  shownPath,
  onShow,
}: {
  result: SolveResponse;
  shownPath: number[] | null;
  onShow: (path: number[] | null) => void;
}) {
  const plan = result.plan;
  const alts = (plan?.alternatives ?? []).filter((a) => a.ok && a.path.length > 1 && a.path.join() !== plan?.path.join());
  if (!plan || alts.length === 0) return null;

  return (
    <div className="alternatives">
      <span>Also try:</span>
      <button className={shownPath === null ? "active" : ""} onClick={() => onShow(null)}>
        the {plan.strategy} line (chosen)
      </button>
      {alts.map((route) => (
        <button
          className={shownPath?.join() === route.path.join() ? "active" : ""}
          key={route.strategy + route.path.join()}
          onClick={() => {
            onShow(route.path);
            document.getElementById("route-image")?.scrollIntoView({ behavior: "smooth", block: "center" });
          }}
        >
          the {route.strategy} line, {route.move_count} moves, {formatInches(route.max_reach_ratio * plan.reach_in)} max reach
        </button>
      ))}
    </div>
  );
}

/* ---------------- the page ---------------- */

export default function App() {
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [detection, setDetection] = useState<DetectResponse | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [round, setRound] = useState(0);
  const [activeHold, setActiveHold] = useState(1);
  const [altPath, setAltPath] = useState<number[] | null>(null);
  const [color, setColor] = useState("");
  const [solvedColor, setSolvedColor] = useState("");
  const [liveTrace, setLiveTrace] = useState<TraceEntry[]>([]);
  const [picks, setPicks] = useState<Picks>({ start: null, finish: null });
  const [picking, setPicking] = useState<"start" | "finish" | null>(null); // on the route, after solving
  const [setupPicking, setSetupPicking] = useState<"start" | "finish" | null>(null); // on the photo, before
  const [hovered, setHovered] = useState<TraceEntry | null>(null);
  const [grips, setGrips] = useState<Grips>({});
  const skipPacing = useRef(false);
  const lastArgs = useRef<ProfileArgs | null>(null);
  const photoId = useRef(0);

  // Forget a color that the new photo doesn't have.
  useEffect(() => {
    if (color && detection && !(color in detection.colors)) setColor("");
  }, [detection, color]);

  const handlePhoto = async (file: File) => {
    const id = ++photoId.current;
    setError(null);
    setResult(null);
    setDetection(null);
    setPicks({ start: null, finish: null });
    setPicking(null);
    setSetupPicking(null);
    setGrips({});
    try {
      const p = await loadPhoto(file);
      if (id !== photoId.current) return;
      setPhoto(p);
      setDetecting(true);
      const det = await detect(p.b64);
      if (id === photoId.current) setDetection(det);
    } catch (e) {
      // The route can still be found without the preview; just no color list.
      if (id === photoId.current) setError(`Couldn't preview the holds: ${(e as Error).message}`);
    } finally {
      if (id === photoId.current) setDetecting(false);
    }
  };

  const findRoute = async (args: ProfileArgs, chosen: Picks = picks) => {
    if (!photo) return;
    const { height, wingspan, level, color } = args;
    lastArgs.current = args;
    setPicks(chosen);
    setPicking(null);
    setSetupPicking(null);
    setHovered(null);
    setSolving(true);
    setError(null);
    setResult(null);
    setLiveTrace([]);
    requestAnimationFrame(() => document.getElementById("route")?.scrollIntoView({ behavior: "smooth" }));

    // The agents can answer faster than anyone reads, so play their messages back at reading speed.
    const pending: TraceEntry[] = [];
    let streamDone = false;
    skipPacing.current = false;
    const player = (async () => {
      for (;;) {
        const next = pending.shift();
        if (next) {
          setLiveTrace((t) => [...t, next]);
          for (let waited = 0; waited < readingTime(next) && !skipPacing.current; waited += 100) await sleep(100);
        } else if (streamDone) return;
        else await sleep(100);
      }
    })();

    const routeGrips = (detection?.holds ?? [])
      .filter((h) => (!color || h.color === color) && grips[holdKey(h)])
      .map((h) => ({ x: h.x, y: h.y, type: grips[holdKey(h)] }));
    try {
      const res = await solvePhotoStream(
        {
          image_b64: photo.b64,
          climber: { height_in: height, wingspan_in: wingspan, level },
          route_color: color || null,
          start_point: chosen.start,
          finish_point: chosen.finish,
          grips: routeGrips,
        },
        (entry) => pending.push(entry),
      ).finally(() => {
        streamDone = true;
      });
      await player;
      setResult(res);
      setHovered(null);
      setSolvedColor(color);
      setAltPath(null);
      setActiveHold(1);
      if (!res.ok) setError(res.status);
      requestAnimationFrame(() => document.getElementById("route")?.scrollIntoView({ behavior: "smooth" }));
    } catch (e) {
      await player;
      setError((e as Error).message);
    } finally {
      setSolving(false);
    }
  };

  const attempts = useMemo(() => (result ? buildAttempts(result) : []), [result]);
  useEffect(() => setRound(Math.max(0, attempts.length - 1)), [attempts]);

  const isFinal = round === attempts.length - 1;
  const baseAttempt = attempts[round];
  const attempt: Attempt | undefined =
    baseAttempt && isFinal && altPath ? { ...baseAttempt, path: altPath, rejected: [] } : baseAttempt;
  const moveTypes = useMemo(() => {
    if (!result || !attempt) return [];
    const byId = new Map(result.holds.map((h) => [h.id, h.type]));
    return attempt.path.slice(1).map((id) => byId.get(id) ?? "unknown");
  }, [result, attempt]);

  const diff = result?.difficulty;
  const footer = result
    ? [`${result.holds.length} holds`, `${result.rounds} round${result.rounds === 1 ? "" : "s"}`, result.critic_approved ? "Critic approved" : null, `run by ${result.orchestrator_source}`]
        .filter(Boolean)
        .join(" · ")
    : "";

  // Picks are holds of the chosen color; a new color means picking again.
  const changeColor = (c: string) => {
    setColor(c);
    setPicks({ start: null, finish: null });
    setSetupPicking(null);
  };
  const pickOnPhoto = (h: Hold) => {
    if (!setupPicking) return;
    const other = setupPicking === "start" ? "finish" : "start";
    const same = picks[other]?.[0] === h.x && picks[other]?.[1] === h.y;
    setPicks({ ...picks, [setupPicking]: [h.x, h.y], ...(same ? { [other]: null } : {}) });
    setSetupPicking(null);
  };

  // While the agents work, the wall shows what the latest message is about (or the one you hover).
  const liveSketch = [...liveTrace].reverse().find((e) => sketchOf(e));
  const shownLive = (hovered && sketchOf(hovered) ? hovered : null) ?? liveSketch;
  const resultSketch = hovered && sketchOf(hovered) ? { key: hovered.step, sketch: sketchOf(hovered)! } : null;

  return (
    <div id="top">
      <Header />
      <HealthBanner />
      <main>
        <section className="intro" id="wall">
          <div className="intro-copy">
            <h1>Beta that fits your reach.</h1>
            <p>Add a photo of the wall and tell us how you climb. We&apos;ll find a route and explain every move.</p>
          </div>
          <div className="wall-setup">
            <WallUpload
              photo={photo}
              detection={detection}
              detecting={detecting}
              color={color}
              picks={picks}
              picking={setupPicking}
              onPick={pickOnPhoto}
              grips={grips}
              setGrip={(h, g) =>
                setGrips((prev) => {
                  const next = { ...prev };
                  if (g) next[holdKey(h)] = g;
                  else delete next[holdKey(h)];
                  return next;
                })
              }
              onPhoto={handlePhoto}
            />
            <ProfileForm
              colors={detection?.colors ?? {}}
              color={color}
              setColor={changeColor}
              pickBar={
                <PickBar
                  before
                  disabled={!detection}
                  picks={picks}
                  picking={setupPicking}
                  setPicking={setSetupPicking}
                  onAuto={() => setPicks({ start: null, finish: null })}
                />
              }
              canSubmit={!!photo}
              solving={solving}
              error={error}
              onFind={findRoute}
            />
          </div>
        </section>

        <section className="route-section" id="route">
          <div className="section-heading">
            <h2>
              Your route
              {diff && (
                <span className="grade-badge" title={diff.reasoning}>
                  {diff.grade} · {diff.label}
                </span>
              )}
            </h2>
            <p>
              {result?.ok
                ? "The orange line is the path. Choose a numbered hold to see its move below."
                : "Your route will show up here once the agents agree on one."}
            </p>
          </div>
          {result && photo && attempt ? (
            <div className="route-layout">
              <div className="wall-side" id="route-image">
                <RouteWall
                  photo={photo}
                  holds={result.holds}
                  color={solvedColor}
                  attempt={attempt}
                  moveTypes={moveTypes}
                  activeHold={activeHold}
                  setActiveHold={setActiveHold}
                  picking={picking}
                  sketch={resultSketch}
                  onPick={(h) => {
                    if (!lastArgs.current || !picking) return;
                    findRoute(lastArgs.current, { ...picks, [picking]: [h.x, h.y] });
                  }}
                />
                <RoundTabs attempts={attempts} round={round} setRound={setRound} />
                <PickBar
                  picks={picks}
                  picking={picking}
                  setPicking={(mode) => {
                    setPicking(mode);
                    if (mode) {
                      setAltPath(null);
                      setRound(attempts.length - 1);
                    }
                  }}
                  onAuto={() => lastArgs.current && findRoute(lastArgs.current, { start: null, finish: null })}
                />
              </div>
              <Conversation
                trace={result.trace}
                upTo={attempt.traceEnd}
                footer={footer}
                drawnStep={resultSketch?.key}
                onHover={setHovered}
              />
            </div>
          ) : solving && photo ? (
            <div className="route-layout">
              <div className="wall-side">
                <WallStage photo={photo} className="route-wall planning sketching">
                  {shownLive && <SketchLayer key={shownLive.step} sketch={sketchOf(shownLive)!} photo={photo} />}
                  {detection?.holds
                    .filter((h) => !color || h.color === color)
                    .map((h) => (
                      <span
                        key={h.id}
                        className="detected-hold faint"
                        style={{ left: pct(h.x, photo.width), top: pct(h.y, photo.height) }}
                      />
                    ))}
                </WallStage>
                <div className="round-tabs">
                  <span>{workingAgent(liveTrace)} is working…</span>
                  <button className="skip-ahead" onClick={() => (skipPacing.current = true)}>
                    Skip ahead
                  </button>
                </div>
              </div>
              <Conversation
                trace={liveTrace}
                upTo={liveTrace.length}
                footer=""
                live
                drawnStep={shownLive?.step}
                onHover={setHovered}
              />
            </div>
          ) : (
            <div className="route-empty">No route yet.</div>
          )}
        </section>

        {result?.ok && (
          <section className="beta-section" id="beta">
            <div className="section-heading beta-heading">
              <h2>How to climb it</h2>
              <p>{result.beta?.summary ?? "Take your time. Tap any move to find it on the wall."}</p>
            </div>
            <MoveList result={result} activeHold={activeHold} setActiveHold={setActiveHold} />
            {!!result.beta?.tips.length && (
              <ul className="tips">
                {result.beta.tips.map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            )}
            <Alternatives
              result={result}
              shownPath={altPath}
              onShow={(p) => {
                setAltPath(p);
                setRound(attempts.length - 1);
              }}
            />
          </section>
        )}
      </main>
      <footer>
        <span>Beta Agents</span>
        <p>Check the landing, climb within your limits, and have a good session.</p>
      </footer>
    </div>
  );
}
