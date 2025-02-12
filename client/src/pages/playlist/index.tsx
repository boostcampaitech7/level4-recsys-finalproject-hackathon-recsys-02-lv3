import { useMutation, useQuery } from "@tanstack/react-query";
import { invariant } from "es-toolkit";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { CandidatesPageContent } from "~/components/CandidatesPageContent";
import { FullScreenLoader } from "~/components/FullScreenLoader";
import { playlistTracksQuery, postTrackMutation } from "~/remotes";
import { PostTrackRequest } from "~/remotes/dio";
import { useUserId } from "~/utils/userInfoContext";

export const Component = () => {
  const navigate = useNavigate();
  const { playlistId } = useParams<{ playlistId: string }>();
  const [searchParams] = useSearchParams();
  const playlistName = searchParams.get("name");

  invariant(playlistId, "undefined playlistId");
  invariant(playlistName, "undefined playlistName");

  const id = useUserId();
  const { data, isLoading } = useQuery(
    playlistTracksQuery(playlistId, id, playlistName)
  );

  const { mutateAsync } = useMutation(postTrackMutation(playlistId, id));
  const handleSubmit = async (payload: PostTrackRequest[]) => {
    await mutateAsync({
      items: payload,
    });
    navigate("/home");
  };

  if (isLoading) {
    return <FullScreenLoader />;
  }
  return <CandidatesPageContent data={data ?? []} onSubmit={handleSubmit} />;
};
