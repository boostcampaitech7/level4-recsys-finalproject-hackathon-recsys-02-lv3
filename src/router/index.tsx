import { Outlet, createBrowserRouter } from "react-router-dom";
import { Layout } from "./Layout";

export const ROUTES = [
  { path: "", lazy: () => import("~/pages/landing") },
  { path: "authorized", lazy: () => import("~/pages/authorized") },
  { path: "onboarding", lazy: () => import("~/pages/onboarding") },
  { path: "home", lazy: () => import("~/pages/home") },
  { path: "playlist/:playlistId", lazy: () => import("~/pages/playlist") },
];

export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Layout>
        <Outlet />
      </Layout>
    ),
    children: ROUTES,
  },
]);
