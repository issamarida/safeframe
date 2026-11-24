import type { CaptureContext } from "../api/types";

/** Mirrors context.GUARDED_KINDS. */
const KIND_TEXT: Record<string, string> = {
  bank: "a bank",
  atm: "an ATM",
  jeweller: "a jeweller",
  pharmacy: "a pharmacy",
  casino: "a casino",
  court: "a court",
  police: "a police station",
  prison: "a prison",
  airport: "an airport",
  border: "a border crossing",
  government: "a government building",
};

/** Which rules apply to this capture, and why. */
export function ContextBanner({ context }: { context?: Partial<CaptureContext> }) {
  if (context?.guarded) {
    const where =
      context.guarded_source === "registry"
        ? `${context.zone_name ?? "a guarded place"}${
            context.zone_kind ? ` (${KIND_TEXT[context.zone_kind] ?? context.zone_kind})` : ""
          }, from the location recorded in the file`
        : "a guarded place, as you declared at upload";
    return (
      <div className="panel context guarded" role="status">
        <strong>Guarded place</strong>
        <p className="small">
          Taken at {where}. Here a band protects only someone who looks under 18: a band is a
          privacy request, not a disguise. Adults wearing one stay visible
          {context.protect_minors ? ", and event mode still blurs likely minors" : ""}.
        </p>
      </div>
    );
  }
  if (context?.protect_minors) {
    return (
      <div className="panel context event" role="status">
        <strong>Event mode</strong>
        <p className="small">
          Everyone wearing a band is blurred, and so is anyone who may be under 18, band or not.
        </p>
      </div>
    );
  }
  return (
    <div className="panel context" role="status">
      <strong>Standard rules</strong>
      <p className="small">
        Everyone wearing a SafeFrame Band is blurred; everyone else stays visible. Nobody’s age is
        estimated{context?.had_location ? ", and the location in the file is not a guarded place" : ""}.
      </p>
    </div>
  );
}
