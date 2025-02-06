import { css } from "@emotion/react";
import AddIcon from "~/assets/svg/AddCircle.svg";
import RemoveIcon from "~/assets/svg/DeleteCircle.svg";
import { Spacing } from "~/components/Spacing";

export const OcrItem = ({
  trackName,
  artistName,
  onSelectChange,
  selected,
}: {
  trackName: string;
  artistName: string;
  onSelectChange: () => void;
  selected: boolean;
}) => {
  return (
    <>
      <Spacing size={10} />
      <div
        css={css({
          display: "flex",
          justifyContent: "space-between",
          width: "100%",
          padding: "5px",
        })}
      >
        <div css={css({ display: "flex" })}>
          <div
            css={css({
              display: "flex",
              flexDirection: "column",
              padding: "5px 15px",
            })}
          >
            <span>{trackName}</span>
            <span css={css({ fontSize: 12 })}>{artistName}</span>
          </div>
        </div>
        <img src={selected ? AddIcon : RemoveIcon} onClick={onSelectChange} />
      </div>
      <Spacing size={10} />
    </>
  );
};
