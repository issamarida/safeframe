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
  notice: "Age estimation is probabilistic.",
  intended_use: "SafeFrame is intended to protect children's privacy in video you share.",
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
    expect(screen.getByText(/deleted as soon as the blurred video is ready/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox"));
    expect(screen.getByText(/after 24 h of inactivity/i)).toBeInTheDocument();
  });
});

describe("Notice", () => {
  it("always states that age estimation is probabilistic", async () => {
    render(<Notice config={config} />);
    expect(screen.getByRole("note")).toHaveTextContent(/age estimation is probabilistic/i);
    await userEvent.click(screen.getByRole("button", { name: /more/i }));
    expect(screen.getByText(config.intended_use)).toBeInTheDocument();
  });
});
