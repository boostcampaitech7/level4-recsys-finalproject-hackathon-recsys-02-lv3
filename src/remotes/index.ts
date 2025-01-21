import { api } from "~/libs/api";
import { queryOptions } from "~/libs/react-query";

export interface PlaylistResponse {
  message: string;
  items: PlayListSchema[];
}
interface PlayListSchema {
  playlist_id: string;
  playlist_name: string;
  playlist_img_url: string;
}

const PLAYLIST_URL = (id: number) => `/users/${id}/playlists`;

export const playlistQuery = (id: number) =>
  queryOptions({
    queryKey: [PLAYLIST_URL(id)],
    queryFn: async () => await api.get<PlaylistResponse>(PLAYLIST_URL(id)),
  });
