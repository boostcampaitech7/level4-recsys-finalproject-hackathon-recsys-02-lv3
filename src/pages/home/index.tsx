import { css } from "@emotion/react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Button } from "~/components/Button";
import { Description } from "~/components/Description";
import { MobilePadding } from "~/components/MobilePadding";
import { ProfileImage } from "~/components/ProfileImage";
import { Spacing } from "~/components/Spacing";
import { Title } from "~/components/Title";
import { playlistQuery } from "~/remotes";
import { useUserId } from "~/utils/userInfoContext";

export const Component = () => {
  const navigate = useNavigate();
  const userId = useUserId();
  const { data } = useQuery(playlistQuery(userId));

  return (
    <>
      <Spacing size={40} />
      <MobilePadding>
        <div
          css={css({
            display: "flex",
            width: "100%",
            overflowX: "auto",
            alignItems: "center",
            gap: 18,
          })}
        >
          <ProfileImage />
          <Title>자영업자를 위한 플레이리스트 추천</Title>
        </div>
        <Spacing size={40} />
        <Description>플레이리스트를 선택해주세요</Description>
        <Spacing size={20} />
      </MobilePadding>
      <div
        css={css({
          display: "flex",
          width: "100%",
          overflowX: "auto",
        })}
      >
        {data?.items.map((v) => (
          <div
            key={v.playlist_id}
            onClick={() =>
              navigate(`/playlist/${v.playlist_id}?name=${v.playlist_name}`)
            }
            css={css({ marginLeft: 20, "&:last-of-type": { marginRight: 20 } })}
          >
            <SquareImage size={150} imageUrl={v.playlist_img_url} />
            {v.playlist_name}
          </div>
        ))}
      </div>
      <Spacing size={80} />
      <MobilePadding>
        <Title>플레이리스트가 없다면</Title>
        <Spacing size={20} />
        <Button backgroundColor="#5b52ff" onClick={() => navigate("/ocr")}>
          사진으로 외부 플레이리스트 불러오기
        </Button>
      </MobilePadding>
    </>
  );
};

const SquareImage = ({
  size,
  imageUrl,
}: {
  size: number;
  imageUrl?: string;
}) => {
  return (
    <div
      css={css({
        width: size,
        height: size,
        overflow: "hidden",
        display: "flex",
        cursor: "pointer",
        alignItems: "center",
        justifyContent: "center",
      })}
    >
      <img
        src={imageUrl}
        css={css({
          width: "100%",
          height: "100%",
          objectFit: "cover",
          objectPosition: "center",
        })}
      />
    </div>
  );
};
