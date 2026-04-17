module.exports = {
  apps: [{
    name: 'options-predictor',
    cwd: 'C:/josho-trader',
    script: 'python',
    args: 'options_predictor.py',
    watch: false,
    autorestart: true,
    max_restarts: 5,
    restart_delay: 60000,
    env: {
      NODE_ENV: 'production',
    },
    // Log management
    log_file: 'C:/josho-trader/logs/predictor-pm2.log',
    error_file: 'C:/josho-trader/logs/predictor-pm2-error.log',
    out_file: 'C:/josho-trader/logs/predictor-pm2-out.log',
    merge_logs: true,
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    // Restart policy
    exp_backoff_restart_delay: 60000,
    max_memory_restart: '500M',
    // Cron: restart daily at 9:10 AM IST (3:40 AM UTC)
    cron_restart: '40 3 * * 1-5',
  }],
};
