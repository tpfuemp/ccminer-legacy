/* SPDX-License-Identifier: GPL-3.0-or-later */
/* REST API route table, also the source of the GET /api/v1/ capability list */

#ifndef API_ROUTES_H
#define API_ROUTES_H

#include "api_http.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-connection handler context. /quit only sets quit_requested, the caller
 * exits after the response is sent. */
typedef struct {
	bool quit_requested;
} api_ctx;

/* The table, and its length. */
const api_route *api_routes_get(size_t *count);

/* Serialized "miner" object for the transport's error envelope. Caller frees. */
char *api_routes_miner_json_str(void);

#ifdef __cplusplus
}
#endif

#endif /* API_ROUTES_H */
