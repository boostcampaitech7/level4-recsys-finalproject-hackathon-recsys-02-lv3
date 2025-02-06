import { css } from "@emotion/react";
import { Description } from "./Description";
import { MobilePadding } from "./MobilePadding";
import { RefreshButton } from "./RefreshButton";
import { Spacing } from "./Spacing";
import { Title } from "./Title";
import { TrackItem } from "./TrackItem";
import { FixedButton } from "./Button";
import { PostTrackRequest, TrackSchema } from "~/remotes/dio";
import { useState } from "react";

const PAGE_SIZE = 10;

export const CandidatesPageContent = ({
  data,
  onSubmit,
}: {
  data: TrackSchema[];
  onSubmit: (payload: PostTrackRequest[]) => Promise<void>;
}) => {
  const [selected, setSelected] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState<number>(0);
  const currentTrack = data.slice(
    currentPage * PAGE_SIZE,
    (currentPage + 1) * PAGE_SIZE
  );

  const handleSelectChange = (trackId: string) => {
    setSelected((prevSelected) => {
      if (prevSelected.includes(trackId)) {
        return prevSelected.filter((item) => item !== trackId);
      } else {
        return [...prevSelected, trackId];
      }
    });
  };

  const handleNextPage = () => {
    if ((currentPage + 1) * PAGE_SIZE < data.length) {
      setCurrentPage((prev) => prev + 1);
    }
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
      <Description>
        '플레이리스트 완성하기'를 누르면
        <br />내 스포티파이 플레이리스트에 트랙이 추가돼요
      </Description>
      <Spacing size={40} />
      <RefreshButton
        css={css({ marginLeft: "auto" })}
        onClick={handleNextPage}
        disabled={(currentPage + 1) * PAGE_SIZE >= data.length}
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
      <FixedButton onClick={() => onSubmit(getPayload(data, selected))}>
        플레이리스트 완성하기
      </FixedButton>
      <Spacing size={20} />
    </MobilePadding>
  );
};

const getPayload = (
  data: TrackSchema[],
  selected: string[]
): PostTrackRequest[] => {
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
