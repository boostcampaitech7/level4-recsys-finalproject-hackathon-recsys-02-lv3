import { css } from "@emotion/react";
import { ComponentProps } from "react";

const tagStyle = (isSelected: boolean) =>
  css({
    padding: "0px 10px",
    height: 40,
    display: "inline-block",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 400,
    fontSize: 15,
    borderRadius: 16,
    borderWidth: 1,
    backgroundColor: isSelected ? "#e5e5e5" : "#3d3d3d",
    marginRight: 6,
    marginBottom: 8,
    color: isSelected ? "#2c2c2c" : "#ededed",
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
