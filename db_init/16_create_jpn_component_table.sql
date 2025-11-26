--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5 (Debian 17.5-1.pgdg110+1)
-- Dumped by pg_dump version 17.5 (Debian 17.5-1.pgdg110+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: jpn_components; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.jpn_components (
    seq bigint,
    component bigint,
    node bigint
);


ALTER TABLE public.jpn_components OWNER TO postgres;

--
-- PostgreSQL database dump complete
--

-- thêm data
INSERT INTO public.jpn_components (seq, component, node)
SELECT row_number() OVER () AS seq, component, node
FROM pgr_connectedComponents(
    'SELECT gid AS id, source, target, cost_s AS cost, reverse_cost FROM jpn_ways'
);

CREATE INDEX idx_jpn_components_node ON public.jpn_components (node);
CREATE INDEX idx_jpn_components_component ON public.jpn_components (component);