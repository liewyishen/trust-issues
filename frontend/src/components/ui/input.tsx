import * as React from "react"

import { cn } from "@/lib/utils"

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type, ...props }, ref) => (
  <input
    type={type}
    ref={ref}
    className={cn(
      "flex h-9 w-full rounded-md border border-border bg-surface-2 px-3 text-sm text-fg",
      "placeholder:text-faint focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-border-strong",
      "disabled:cursor-not-allowed disabled:opacity-50",
      // Numeric fields are compared against one another by eye; align the figures.
      type === "number" && "tnum",
      // The spinners add nothing and steal width from the figures.
      "[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none",
      className,
    )}
    {...props}
  />
))
Input.displayName = "Input"
