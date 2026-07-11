import { isQwenCombinerConfigured } from "@/lib/server/qwenCombiner";
import { NextResponse } from "next/server";

// Public runtime flags the client needs before the first render. Currently just
// whether the fine-tuned Qwen combiner is on offer, so the app can default to it.
export function GET() {
  return NextResponse.json({ qwenAvailable: isQwenCombinerConfigured() });
}
