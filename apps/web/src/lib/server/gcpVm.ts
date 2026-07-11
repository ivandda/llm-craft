import { getGoogleAccessToken } from "@/lib/server/vertexCombiner";

const COMPUTE_ENDPOINT = "https://compute.googleapis.com/compute/v1";
const REQUEST_TIMEOUT_MS = 15_000;

export type VmStatus = {
  name: string;
  zone: string;
  project: string;
  machineType: string;
  accelerator?: string;
  // RUNNING | TERMINATED | STOPPING | STAGING | PROVISIONING | SUSPENDED ...
  status: string;
  internalIp?: string;
  externalIp?: string;
  lastStartTimestamp?: string;
  lastStopTimestamp?: string;
  consoleUrl: string;
};

type ComputeInstance = {
  name?: string;
  status?: string;
  machineType?: string;
  guestAccelerators?: Array<{ acceleratorType?: string; acceleratorCount?: number }>;
  networkInterfaces?: Array<{
    networkIP?: string;
    accessConfigs?: Array<{ natIP?: string }>;
  }>;
  lastStartTimestamp?: string;
  lastStopTimestamp?: string;
};

export class GcpVmError extends Error {}

export function getVmTarget() {
  const project = process.env.GOOGLE_CLOUD_PROJECT?.trim();

  if (!project) {
    throw new GcpVmError("GOOGLE_CLOUD_PROJECT is not configured");
  }

  return {
    project,
    zone: process.env.QWEN_VM_ZONE?.trim() || "us-central1-a",
    name: process.env.QWEN_VM_NAME?.trim() || "qwen-combiner-test"
  };
}

export async function getVmStatus(): Promise<VmStatus> {
  const target = getVmTarget();
  const instance = (await computeRequest(
    `${instanceUrl(target)}`,
    "GET"
  )) as ComputeInstance;

  return {
    name: instance.name ?? target.name,
    zone: target.zone,
    project: target.project,
    machineType: lastSegment(instance.machineType),
    accelerator: instance.guestAccelerators?.[0]
      ? `${instance.guestAccelerators[0].acceleratorCount ?? 1}x ${lastSegment(
          instance.guestAccelerators[0].acceleratorType
        )}`
      : undefined,
    status: instance.status ?? "UNKNOWN",
    internalIp: instance.networkInterfaces?.[0]?.networkIP,
    externalIp: instance.networkInterfaces?.[0]?.accessConfigs?.[0]?.natIP,
    lastStartTimestamp: instance.lastStartTimestamp,
    lastStopTimestamp: instance.lastStopTimestamp,
    consoleUrl: `https://console.cloud.google.com/compute/instancesDetail/zones/${target.zone}/instances/${target.name}?project=${target.project}`
  };
}

export async function startVm(): Promise<void> {
  await computeRequest(`${instanceUrl(getVmTarget())}/start`, "POST");
}

export async function stopVm(): Promise<void> {
  await computeRequest(`${instanceUrl(getVmTarget())}/stop`, "POST");
}

function instanceUrl(target: { project: string; zone: string; name: string }): string {
  return `${COMPUTE_ENDPOINT}/projects/${target.project}/zones/${target.zone}/instances/${target.name}`;
}

async function computeRequest(url: string, method: "GET" | "POST"): Promise<unknown> {
  const accessToken = await getGoogleAccessToken();
  const response = await fetch(url, {
    method,
    headers: { Authorization: `Bearer ${accessToken}` },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  }).catch((error: unknown) => {
    throw new GcpVmError(error instanceof Error ? error.message : String(error));
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new GcpVmError(
      `Compute API request failed with ${response.status}${detail ? `: ${detail.slice(0, 300)}` : ""}`
    );
  }

  return response.json();
}

function lastSegment(value?: string): string {
  return value?.split("/").pop() ?? "unknown";
}
