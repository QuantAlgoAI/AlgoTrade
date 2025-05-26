#!/bin/bash
# Backup PostgreSQL database to backups/ directory with timestamp
# Usage: bash backup_postgres.sh

set -e

BACKUP_DIR="backups"
mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILENAME="$BACKUP_DIR/pg_backup_$TIMESTAMP.sql"

# Load environment variables
export $(grep -v '^#' .env | xargs)

pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "$FILENAME"
echo "Backup saved to $FILENAME" 