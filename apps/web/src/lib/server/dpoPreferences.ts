import { query } from "@/lib/server/db";
import type { AuthUser, DpoPreferenceRequest, ElementToken } from "@/lib/types";
import { randomBytes } from "crypto";

export type DpoPreferenceEvent = {
  id: string;
  userId: string;
  mode: string;
  goalId?: string;
  inputA: ElementToken;
  inputB: ElementToken;
  shownOutputs: ElementToken[];
  selectedOutput: ElementToken;
  rejectedOutputs: ElementToken[];
  inventorySnapshot: ElementToken[];
  combinationIndex: number;
  source: string;
  createdAt: string;
};

type DpoPreferenceRow = {
  id: string;
  user_id: string;
  mode: string;
  goal_id: string | null;
  input_a: ElementToken;
  input_b: ElementToken;
  shown_outputs: ElementToken[];
  selected_output: ElementToken;
  rejected_outputs: ElementToken[];
  inventory_snapshot: ElementToken[];
  combination_index: number;
  source: string;
  created_at: Date;
};

export async function saveDpoPreference(input: {
  user: AuthUser;
  preference: DpoPreferenceRequest;
}): Promise<DpoPreferenceEvent> {
  const rejectedOutputs = input.preference.shownOutputs.filter(
    (output) => output.id !== input.preference.selectedOutput.id
  );
  const result = await query<DpoPreferenceRow>(
    `
    INSERT INTO dpo_preference_events (
      id, user_id, mode, goal_id, input_a, input_b, shown_outputs,
      selected_output, rejected_outputs, inventory_snapshot, combination_index,
      source
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    RETURNING id, user_id, mode, goal_id, input_a, input_b, shown_outputs,
      selected_output, rejected_outputs, inventory_snapshot, combination_index,
      source, created_at
    `,
    [
      randomBytes(12).toString("hex"),
      input.user.id,
      input.preference.mode,
      input.preference.goalId ?? null,
      input.preference.inputA,
      input.preference.inputB,
      input.preference.shownOutputs,
      input.preference.selectedOutput,
      rejectedOutputs,
      input.preference.inventorySnapshot,
      input.preference.combinationIndex,
      input.preference.source
    ]
  );

  return toDpoPreferenceEvent(result.rows[0]);
}

function toDpoPreferenceEvent(row: DpoPreferenceRow): DpoPreferenceEvent {
  return {
    id: row.id,
    userId: row.user_id,
    mode: row.mode,
    goalId: row.goal_id ?? undefined,
    inputA: row.input_a,
    inputB: row.input_b,
    shownOutputs: row.shown_outputs,
    selectedOutput: row.selected_output,
    rejectedOutputs: row.rejected_outputs,
    inventorySnapshot: row.inventory_snapshot,
    combinationIndex: row.combination_index,
    source: row.source,
    createdAt: row.created_at.toISOString()
  };
}
