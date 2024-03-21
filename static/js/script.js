// ! Functions that deal with button events

$(function () {
  // * Model switch
  $("a#use-model").bind("click", function () {
    $.getJSON("/detect", function (data) {
      // do nothing
    });
    return false;
  });
});


$(function () {
  // * reset camera
  $("a#prova").bind("click", function () {
    $.getJSON("/prova_prova", function (data) {
      // do nothing
    });
    return false;
  });
});
