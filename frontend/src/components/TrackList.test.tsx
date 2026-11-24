import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { TrackData } from "../api/types";
import { indexTracks, type Policy } from "../review/model";
import { ContextBanner } from "./ContextBanner";
import { TrackList } from "./TrackList";

const policy: Policy = {
  threshold: 0.35,
  minFaceSamples: 3,
  minBandHits: 1,
  guarded: false,
  protectMinors: false,
};

function person(id: number, sightings: number, p: number | null = null, faces = 0): TrackData {
  return {
    track_id: id,
    first_frame: 0,
    last_frame: 50,
    p_minor: p,
    expected_age: null,
    face_samples: faces,
    peak_confidence: 0.9,
    boxes: [[0, 0, 0, 10, 10, 0]],
    band: {
      confirmed: sightings,
      possible: 0,
      first_frame: sightings ? 25 : null,
      via: { wrist: sightings, body: 0, box: 0 },
      hits: [],
    },
  };
}

function renderList(p: Policy, onOverride = vi.fn(), onJump = vi.fn()) {
  render(
    <TrackList
      thumbnailUrl={() => ""}
      fps={25}
      tracks={indexTracks([person(1, 4, 0.05, 3), person(2, 0)])}
      policy={p}
      overrides={{}}
      selected={null}
      reveal={false}
      onSelect={vi.fn()}
      onJump={onJump}
      onOverride={onOverride}
    />,
  );
  return { onOverride, onJump };
}

describe("TrackList", () => {
  it("labels wearers and bystanders, without age details outside guarded zones", async () => {
    const { onJump } = renderList(policy);
    expect(screen.getByText("Wearing a band")).toBeInTheDocument();
    expect(screen.getByText("No band")).toBeInTheDocument();
    expect(screen.queryByText(/P\(<18\)/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /first at 0:01.00/ }));
    expect(onJump).toHaveBeenCalledWith(25);
  });

  it("shows why an adult's band isn't honoured in a guarded place", () => {
    renderList({ ...policy, guarded: true });
    expect(screen.getByText("Band not honoured")).toBeInTheDocument();
    expect(screen.getByText(/P\(<18\) 5%/)).toBeInTheDocument();
  });

  it("asks for consent before keeping a wearer visible", async () => {
    const { onOverride } = renderList(policy);
    const keep = screen.getAllByRole("radio", { name: "Keep" });
    await userEvent.click(keep[0]!);
    expect(onOverride).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/asked not to be shown/);
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    await userEvent.click(keep[0]!);
    await userEvent.click(screen.getByRole("button", { name: /keep anyway/i }));
    expect(onOverride).toHaveBeenCalledWith(1, "keep");

    // Bystanders can be kept (or blurred) directly.
    await userEvent.click(keep[1]!);
    expect(onOverride).toHaveBeenLastCalledWith(2, "keep");
  });
});

describe("ContextBanner", () => {
  it("names the guarded place the location matched", () => {
    render(
      <ContextBanner
        context={{ guarded: true, guarded_source: "registry", zone_name: "Banco de España", zone_kind: "bank" }}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(/Banco de España \(a bank\)/);
    expect(screen.getByRole("status")).toHaveTextContent(/not a disguise/);
  });

  it("explains the standard rules", () => {
    render(<ContextBanner context={{ guarded: false }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/Nobody’s age is estimated/);
  });
});
