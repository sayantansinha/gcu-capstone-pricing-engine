# Provision the S3 buckets
resource "aws_s3_bucket" "pli_buckets" {
  count  = length(local.buckets_all)
  bucket = local.buckets_all[count.index]
  tags   = local.tags
}
