import { css } from "@emotion/react";
import { rgba } from "emotion-rgba";
import { ComponentProps } from "react";

// https://emotion.sh/docs/css-prop#use-the-css-prop
const tagStyle = (isSelected: boolean) =>
  css({
    width: "fit-content",
    margin: "0 auto",
    height: 40,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 600,
    border: "1px solid",
    borderRadius: 7,
    borderColor: "#1ED760",
    backgroundColor: isSelected ? "#1ED760" : rgba("#C3FFC3", 0.75),
  });

export const Tag = ({
  children,
  isSelected,
  ...props
}: ComponentProps<"button"> & { isSelected: boolean }) => {
  return (
    <button {...props} css={tagStyle(isSelected)}>
      {children || null}
    </button>
  );
};
