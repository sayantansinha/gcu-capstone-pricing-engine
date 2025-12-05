resource "aws_s3_object" "rbac_json" {
  bucket = local.security_bucket
  key    = "rbac.json"
  source = "${path.module}/asset/rbac.json"
  etag   = filemd5("${path.module}/asset/rbac.json")
}

resource "aws_s3_object" "users_json" {
  bucket = local.security_bucket
  key    = "users.json"
  source = "${path.module}/asset/users.json"
  etag   = filemd5("${path.module}/asset/users.json")
}