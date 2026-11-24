import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { AppConfig } from "../api/types";
import { Notice } from "./Notice";
import { UploadScreen } from "./UploadScreen";

const config: AppConfig = {
  version: "0.1.0",
  max_upload_mb: 1,
  job_ttl_hours: 24,
  default_threshold: 0.35,
  default_min_face_samples: 3,
  default_min_band_hits: 1,
  sealing_enabled: false,
  seal_retention_days: 30,
  band_spec_version: 1,
  notice: "SafeFrame blurs everyone wearing a SafeFrame Band.",
  intended_use: "The band is a privacy request, not a disguise.",
};

describe("UploadScreen", () => {
  it("requires a file before uploading", () => {
    render(<UploadScreen config={config} onUploaded={vi.fn()} />);
    expect(screen.getByRole("button", { name: /upload and analyse/i })).toBeDisabled();
  });

  it("rejects files over the size limit", async () => {
    render(<UploadScreen config={config} onUploaded={vi.fn()} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const big = new File([new Uint8Array(2 * 1024 * 1024)], "big.mp4", { type: "video/mp4" });
    await userEvent.upload(input, big);
    expect(screen.getByText(/larger than the 1 MB limit/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /upload and analyse/i })).toBeDisabled();
  });

  it("explains retention for both choices", async () => {
    render(<UploadScreen config={config} onUploaded={vi.fn()} />);
    expect(screen.getByText(/deleted as soon as the blurred file is ready/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox", { name: /keep my original/i }));
    expect(screen.getByText(/after 24 h of inactivity/i)).toBeInTheDocument();
  });

  it("accepts photos and links to a printable band", () => {
    render(<UploadScreen config={config} onUploaded={vi.fn()} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input.accept).toContain("image/jpeg");
    expect(screen.getByRole("link", { name: /print a band/i })).toHaveAttribute("href", "/api/band.svg");
  });

  it("sends the guarded and event-mode choices with the upload", async () => {
    const sent: string[] = [];
    class FakeXHR {
      upload = {};
      status = 201;
      responseText = "{}";
      onload: (() => void) | null = null;
      open(_: string, url: string) {
        sent.push(url);
      }
      setRequestHeader() {}
      send() {
        this.onload?.();
      }
      abort() {}
    }
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    const onUploaded = vi.fn();
    render(<UploadScreen config={config} onUploaded={onUploaded} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "bank.jpg", { type: "image/jpeg" }));
    await userEvent.click(screen.getByRole("checkbox", { name: /guarded place/i }));
    await userEvent.click(screen.getByRole("checkbox", { name: /event mode/i }));
    await userEvent.click(screen.getByRole("button", { name: /upload and analyse/i }));
    expect(onUploaded).toHaveBeenCalled();
    const params = new URL(sent[0]!, "http://x").searchParams;
    expect(params.get("guarded")).toBe("true");
    expect(params.get("protect_minors")).toBe("true");
    expect(params.get("filename")).toBe("bank.jpg");
    vi.unstubAllGlobals();
  });
});

describe("Notice", () => {
  it("always states that only a visible band can be honoured", async () => {
    render(<Notice config={config} />);
    expect(screen.getByRole("note")).toHaveTextContent(/only honour a band it can see/i);
    await userEvent.click(screen.getByRole("button", { name: /more/i }));
    expect(screen.getByText(config.intended_use)).toBeInTheDocument();
  });
});
