import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useUserId } from "~/utils/userInfoContext";
import { useMutation, useQuery } from "@tanstack/react-query";
import { playlistTracksQuery, postTrackMutation } from "~/remotes";
import { AuthGuard } from "~/components/AuthGuard";
import { useState } from "react";
import { css } from "@emotion/react";
import { TrackItem } from "~/components/TrackItem";
import { FullScreenLoader } from "~/components/FullScreenLoader";
import { Button, FixedButton } from "~/components/Button";
import { Spacing } from "~/components/Spacing";
import { RefreshButton } from "~/components/RefreshButton";
import { invariant } from "es-toolkit";
import { PostTrackRequest } from "~/remotes/dio";
import { MobilePadding } from "~/components/MobilePadding";
import { Title } from "~/components/Title";

const PlaylistPage = () => {
  const navigate = useNavigate();
  const { playlistId } = useParams<{ playlistId: string }>();
  const [searchParams] = useSearchParams();
  const playlistName = searchParams.get("name");

  invariant(playlistId, "undefined playlistId");
  invariant(playlistName, "undefined playlistName");

  const id = useUserId();
  const [selected, setSelected] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState<number>(0);
  const { data = [], isLoading } = useQuery(
    playlistTracksQuery(playlistId, id, playlistName)
  );
  const { mutateAsync } = useMutation(postTrackMutation(playlistId, id));

  if (isLoading) {
    return <FullScreenLoader />;
  }

  const itemsPerPage = 10;

  const currentTrack = data.slice(
    currentPage * itemsPerPage,
    (currentPage + 1) * itemsPerPage
  );

  const handleNextPage = () => {
    if ((currentPage + 1) * itemsPerPage < data.length) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handleSelectChange = (trackId: string) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(trackId)) {
        return prevSelected.filter((item) => item !== trackId);
      } else {
        return [...prevSelected, trackId];
      }
    });
  };

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
    <MobilePadding>
      <Spacing size={40} />
      <Title>
        선택한 플레이리스트와
        <br />
        어울리는 트랙들이에요
      </Title>
      <Spacing size={16} />
      <div css={css({ color: "#cacaca" })}>
        '플레이리스트 완성하기'를 누르면
        <br />내 스포티파이 플레이리스트에 트랙이 추가돼요
      </div>
      <Spacing size={40} />
      <RefreshButton
        css={css({ marginLeft: "auto" })}
        onClick={handleNextPage}
        disabled={(currentPage + 1) * itemsPerPage >= data.length}
      >
        새로운 추천 결과 받기
      </RefreshButton>
      <Spacing size={10} />
      {currentTrack?.map((v) => (
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
      <Spacing size={20} />
      <FixedButton
        onClick={async () => {
          await mutateAsync({
            items: getPayload(),
          });
          navigate("/home");
        }}
      >
        플레이리스트 완성하기
      </FixedButton>
      <Spacing size={20} />
    </MobilePadding>
  );
};

export const SpotifyPlaylist = () => (
  <AuthGuard>
    <PlaylistPage />
  </AuthGuard>
);
