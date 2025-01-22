import { useNavigate, useParams } from "react-router-dom";
import { useUserId } from "~/utils/userInfoContext";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  PostTrackRequest,
  playlistTracksQuery,
  postTrackMutation,
} from "~/remotes";
import { AuthGuard } from "~/components/AuthGuard";
import { useState } from "react";
import { css } from "@emotion/react";
import { TrackItem } from "~/components/TrackItem";
import { FullScreenLoader } from "~/components/FullScreenLoader";
import { Button } from "~/components/Button";
import { Spacing } from "~/components/Spacing";

const PlaylistPage = () => {
  const navigate = useNavigate();
  const { playlistId } = useParams<{ playlistId: string }>();
  if (playlistId == null) return;
  const id = useUserId();

  const { data, isLoading } = useQuery(playlistTracksQuery(playlistId, id));
  const { mutateAsync } = useMutation(postTrackMutation(playlistId, id));

  const [selected, setSelected] = useState<string[]>([]);
  const handleSelectChange = (trackId: string) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(trackId)) {
        return prevSelected.filter((item) => item !== trackId);
      } else {
        return [...prevSelected, trackId];
      }
    });
  };

  if (isLoading) {
    return <FullScreenLoader />;
  }

  const getPayload = (): PostTrackRequest[] => {
    return selected
      .map((v) => {
        const item = data?.find((track) => track.track_id === v);
        if (item == null) {
          return null;
        }
        return { track_name: item?.track_name, artists: item?.artists };
      })
      .filter((v): v is PostTrackRequest => v != null);
  };

  return (
    <>
      <div css={css({ padding: "0px 24px" })}>
        {data?.map((v) => (
          <TrackItem
            key={v.track_id}
            trackImage={v.track_img_url}
            trackName={v.track_name}
            artistName={v.artists
              .map(({ artist_name }) => artist_name)
              .join(", ")}
            onSelectChange={() => handleSelectChange(v.track_id)}
            selected={selected.includes(v.track_id)}
          />
        ))}
      </div>
      <Button
        css={completeCSS}
        onClick={async () => {
          await mutateAsync({
            items: getPayload(),
          });
          navigate("/home");
        }}
      >
        플레이리스트 완성하기
      </Button>
      <Spacing size={20} />
    </>
  );
};

export const Component = () => (
  <AuthGuard>
    <PlaylistPage />
  </AuthGuard>
);

const completeCSS = css({
  width: "calc(100% - 60px)" /* 양쪽에 20px씩 여백 */,
  maxWidth: 600,
  padding: "15px",
  margin: "0 auto",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
});
