import { useState } from "react";
import type { AppConfig } from "../api/types";

/** Always-visible notice of what SafeFrame can and cannot see. */
export function Notice({ config }: { config: AppConfig }) {
  const [open, setOpen] = useState(false);
  return (
    <aside className="notice" role="note" aria-label="About SafeFrame Bands">
      <div className="notice-row">
        <span className="notice-icon" aria-hidden>
          !
        </span>
        <p>
          <strong>SafeFrame can only honour a band it can see.</strong> A band hidden for the whole
          clip, too far away or too blurry is missed. Always review before sharing.{" "}
          <button className="link" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? "Less" : "More"}
          </button>
        </p>
      </div>
      {open && (
        <div className="notice-more">
          <p>{config.notice}</p>
          <p>{config.intended_use}</p>
        </div>
      )}
    </aside>
  );
}
