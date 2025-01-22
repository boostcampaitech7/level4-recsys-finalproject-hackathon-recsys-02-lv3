import { css } from "@emotion/react";
import AddIcon from "~/assets/svg/PlusCircle.svg";
import AddedIcon from "~/assets/svg/CheckCircle.svg";
import { Spacing } from "~/components/Spacing";

export const TrackItem = ({
  trackImage,
  trackName,
  artistName,
  onSelectChange,
  selected,
}: {
  trackImage: string;
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
        })}
      >
        <div css={css({ display: "flex" })}>
          <img src={trackImage} width="60" />
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
        <img src={selected ? AddedIcon : AddIcon} onClick={onSelectChange} />
      </div>
      <Spacing size={10} />
    </>
  );
};
