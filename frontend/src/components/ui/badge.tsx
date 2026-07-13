import { cva, type VariantProps } from "class-variance-authority"
import * as React from "react"

import { cn } from "@/lib/utils"

/*
 * Colour carries meaning and nothing else.
 *
 *   approve / reject -- the decision, and only the decision
 *   adverse          -- a risk-INCREASING SHAP contribution (a reason code)
 *   neutral          -- no meaning; a label that happens to need a chip
 */
const badgeVariants = cva(
  "inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium tracking-wide",
  {
    variants: {
      variant: {
        neutral: "border-border bg-surface-2 text-muted",
        approve: "border-approve-dim bg-approve-dim/40 text-approve",
        reject: "border-reject-dim bg-reject-dim/40 text-reject",
        adverse: "border-transparent bg-transparent text-adverse",
      },
    },
    defaultVariants: { variant: "neutral" },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export const Badge = ({ className, variant, ...props }: BadgeProps) => (
  <span className={cn(badgeVariants({ variant }), className)} {...props} />
)
