import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useUserId } from "~/utils/userInfoContext";
import { AuthGuard } from "~/components/AuthGuard";
import { useMutation } from "@tanstack/react-query";
import {
  postTestOnboardingMutation,
  postTestOnboardingSelectMutation,
} from "~/remotes";
import { SelectTags } from "./SelectTags";
import { SwitchCase } from "@toss/react";
import { SelectTracks } from "./SelectTracks";
import { OnboardingSelectItemType, TrackOnboardingSchema } from "~/remotes/dio";

export type SelectTrackType = [
  TrackOnboardingSchema[],
  TrackOnboardingSchema[]
];

const OnBoardingPage = () => {
  const navigate = useNavigate();
  const id = useUserId();

  const [step, setStep] = useState<"tag" | "track">("tag");
  const [trackList, setTrackList] = useState<SelectTrackType>();
  const { mutateAsync: mutateSelectTag } = useMutation(
    postTestOnboardingMutation
  );
  const { mutateAsync: mutateSelectTrack } = useMutation(
    postTestOnboardingSelectMutation
  );

  const handleSubmitTags = async (selectedTags: string[]) => {
    const result = await mutateSelectTag({ user_id: id, tags: selectedTags });
    setTrackList([result.items1, result.items2]);
    setStep("track");
  };

  const handleSubmitTracks = async (
    selectedTracks: OnboardingSelectItemType[]
  ) => {
    await mutateSelectTrack({ user_id: id, items: selectedTracks });
    navigate("/home");
  };

  return (
    <>
      <SwitchCase
        caseBy={{
          tag: <SelectTags onSubmit={handleSubmitTags} />,
          track: trackList && (
            <SelectTracks trackList={trackList} onSubmit={handleSubmitTracks} />
          ),
        }}
        value={step}
      />
    </>
  );
};

export const Component = () => (
  <AuthGuard>
    <OnBoardingPage />
  </AuthGuard>
);
