import { useParams } from "react-router-dom";
import { SpotifyPlaylist } from "~/pages/playlist/spotify";
import { OcrPlaylist } from "~/pages/playlist/image";
import { AuthGuard } from "~/components/AuthGuard";

const PlaylistPage = () => {
  const { playlistId } = useParams<{ playlistId: string }>();
  return <>{playlistId !== "image" ? <SpotifyPlaylist /> : <OcrPlaylist />}</>;
};
export const Component = () => (
  <AuthGuard>
    <PlaylistPage />
  </AuthGuard>
);
