module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '/venv/trellis/bin/python',
    args: '--port 8093'
  }]
};
