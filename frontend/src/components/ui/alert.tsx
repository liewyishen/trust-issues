import { cva, type VariantProps } from "class-variance-authority"
import * as React from "react"

import { cn } from "@/lib/utils"

const alertVariants = cva("rounded-md border px-4 py-3 text-sm", {
  variants: {
    variant: {
      /** 422 -- the request is malformed. The applicant did nothing wrong; the
       *  client sent something the contract refuses. */
      invalid: "border-reject-dim bg-reject-dim/25 text-fg",
      /** 503 / network -- the service is not ready, or is not there. */
      unavailable: "border-border-strong bg-surface-2 text-fg",
    },
  },
  defaultVariants: { variant: "unavailable" },
})

export interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {}

export const Alert = ({ className, variant, ...props }: AlertProps) => (
  <div role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
)

export const AlertTitle = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h5 className={cn("mb-1 font-medium", className)} {...props} />
)

export const AlertDescription = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("text-muted", className)} {...props} />
)
