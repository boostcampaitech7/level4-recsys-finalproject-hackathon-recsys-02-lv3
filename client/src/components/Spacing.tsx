import { css } from "@emotion/react";

export const Spacing = ({ size }: { size: number }) => {
  return (
    <div
      css={css`
        height: ${size}px;
        width: 100%;
      `}
    ></div>
  );
};
