import { css } from "@emotion/react";
import { ComponentProps, ReactNode } from "react";

export const Description = ({
  children,
  ...props
}: { children: ReactNode } & ComponentProps<"div">) => {
  return (
    <div css={css({ color: "#cacaca", fontSize: 18 })} {...props}>
      {children}
    </div>
  );
};
