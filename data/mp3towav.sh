for i in $1/*.mp3;
  do name=`echo "$i" | cut -d'.' -f1`
  ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 22050 "${name}.wav"
done