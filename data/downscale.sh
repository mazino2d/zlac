for i in origin/320/*.mp3;
  do name=`echo "$i" | cut -d'.' -f1 | cut -d'/' -f3`
  ffmpeg -i "$i" -codec:a libmp3lame -b:a "$1k" "scale/$1/${name}.mp3"
done