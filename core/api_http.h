/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Minimal HTTP/1.1 transport for the REST API, no miner symbols. Routes with
 * available = false answer 501. No keep-alive, pipelining, chunked bodies or
 * compression: every response is Connection: close.
 */

#ifndef API_HTTP_H
#define API_HTTP_H

#include <stddef.h>
#include <stdbool.h>

#include <jansson.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Limits. A request that exceeds one of these is answered, not dropped. */
#define API_HTTP_MAX_REQLINE      2048   /* request line                     */
#define API_HTTP_MAX_HEADER_BYTES 8192   /* all header lines together        */
#define API_HTTP_MAX_HEADER_LINES 32
#define API_HTTP_MAX_BODY         8192
#define API_HTTP_MAX_PATH         512
#define API_HTTP_MAX_QUERY        512
#define API_HTTP_SOCK_TIMEOUT_S   5

typedef enum {
	API_M_NONE = 0,
	API_M_GET,
	API_M_HEAD,
	API_M_POST,
	API_M_OPTIONS,
	API_M_OTHER
} api_method;

typedef enum {
	API_PRIV_READ = 0,
	API_PRIV_WRITE,
	API_PRIV_CONTROL
} api_priv;

typedef struct {
	api_method method;
	char raw_method[16];
	char path[API_HTTP_MAX_PATH];
	char query[API_HTTP_MAX_QUERY];
	char body[API_HTTP_MAX_BODY + 1];
	size_t body_len;
	char auth_bearer[256];       /* token only, empty when absent          */
	char origin[256];
	char ws_key[128];            /* Sec-WebSocket-Key, empty when absent   */
	bool upgrade_websocket;
	bool pretty;                 /* ?pretty=1                              */
	bool has_content_length;
	size_t content_length;
	bool chunked;                /* Transfer-Encoding: chunked -> 501      */
} api_request;

typedef enum {
	API_PARSE_OK = 0,
	API_PARSE_INCOMPLETE,        /* need more bytes, keep reading          */
	API_PARSE_BAD,               /* 400                                    */
	API_PARSE_TOO_LARGE,         /* 413                                    */
	API_PARSE_UNSUPPORTED        /* 501 (chunked body)                     */
} api_parse_status;

/* Handler contract: never touches the socket, returns the HTTP status.
 * On a non-2xx it may fill errmsg, which becomes error.message. */
typedef int (*api_handler_fn)(const api_request *req, void *ctx,
                              json_t **out, char *errmsg, size_t errlen);

/* text body with its own content type; *out is malloc'd, freed by the transport */
typedef int (*api_text_fn)(const api_request *req, void *ctx,
                           char **out, char *errmsg, size_t errlen);

typedef struct {
	api_method method;
	const char *path;            /* exact, or trailing '/' = prefix match  */
	api_priv priv;
	bool available;              /* false -> 501 not_implemented           */
	api_handler_fn handler;
	api_text_fn text_handler;    /* optional; wins over handler when set   */
	const char *content_type;    /* required when text_handler is set      */
} api_route;

/* Everything the transport needs to know about policy, supplied by the
 * caller so this file stays free of miner globals. */
typedef struct {
	const char *token;           /* NULL/empty = no token required         */
	bool cors;                   /* send Access-Control-Allow-Origin       */
	api_priv granted;            /* privilege of this connection's source  */
	bool control_enabled;        /* --api-control                          */
	const char *miner_json;      /* serialized "miner" object, may be NULL  */
} api_http_config;

/* pure; API_PARSE_INCOMPLETE until headers and body are complete */
api_parse_status api_http_parse(const char *buf, size_t len, api_request *req);

/* Percent-decode in place; false on %00 or a malformed escape. */
bool api_http_percent_decode(char *s);

/* Route lookup. Sets *status to 200/404/405/501 and returns the route (or
 * NULL). A path that matches but with the wrong verb is 405, not 404. */
const api_route *api_http_route(const api_route *routes, size_t nroutes,
                                const api_request *req, int *status);

const char *api_http_status_text(int status);
const char *api_http_error_code(int status);

/* Serialize an error body: {"miner":{...},"error":{...}}. Caller frees. */
char *api_http_error_body(const api_http_config *cfg, int status, const char *message);

/* Write a complete response, handling partial sends. Returns 0 on success. */
int api_http_send(int sock, int status, const char *content_type,
                  const char *body, size_t body_len, const api_http_config *cfg);

/* Read, parse, authorise, dispatch and respond on an accepted socket.
 * Returns the HTTP status served, or -1 if nothing could be sent. */
int api_http_serve(int sock, const api_route *routes, size_t nroutes,
                   void *ctx, const api_http_config *cfg);

/* api_http_serve() with bytes already read from the socket */
int api_http_serve_prefixed(int sock, const char *prefix, size_t prefixlen,
                            const api_route *routes, size_t nroutes,
                            void *ctx, const api_http_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* API_HTTP_H */
